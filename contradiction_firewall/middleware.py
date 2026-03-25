"""
FirewallMiddleware -- the main entry point.

Wraps OpenAI and Anthropic API calls with contradiction detection,
repair, and structured audit logging.

Pipeline per turn:
  1. Call LLM to get a draft response
  2. Extract atomic claims from the draft
  3. Retrieve relevant prior claims from memory + ledger
  4. Run multi-judge detection (rule-based, NLI, LLM judge, numeric)
  5. Risk engine aggregates results and decides action:
       ALLOW    -- pass through
       LOG_ONLY -- log and pass through
       REPAIR   -- self-repair loop, re-check, deliver repaired response
       BLOCK    -- return structured block message
       ESCALATE -- return block + trigger human review webhook
  6. Store assistant claims in memory for future turns
  7. Log everything to configured sinks

Usage:
  firewall = FirewallMiddleware(
      provider="openai",
      model="gpt-4o",
      ledger=my_ledger,
  )
  response = firewall.chat(
      system="You are a helpful assistant.",
      messages=[{"role": "user", "content": "..."}],
  )
  print(response.content)
  print(response.firewall_report)
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .extractor import ClaimExtractor
from .ledger import ConstraintLedger
from .logging_layer import FirewallLogger, InMemoryLogSink
from .memory import ConversationMemory
from .models import (
    ActionDecision,
    Claim,
    ContraEvent,
    FirewallResponse,
)
from .repair import RepairLayer
from .retriever import CandidateRetriever
from .risk_engine import RiskConfig, RiskEngine


# ---------------------------------------------------------------------------
# Firewall configuration
# ---------------------------------------------------------------------------

@dataclass
class FirewallConfig:
    """
    Full configuration for the FirewallMiddleware.

    All thresholds use contradiction confidence (0.0-1.0).
    """
    # Threshold knobs
    block_threshold: float = 0.85
    repair_threshold: float = 0.55
    log_threshold: float = 0.30
    max_repair_attempts: int = 2
    memory_window: int = 10

    # Detector enable flags
    use_nli: bool = True
    use_llm_judge: bool = True
    use_numeric_detector: bool = True

    # NLI model
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    nli_device: Optional[str] = None

    # Extractor mode
    use_llm_extractor: bool = False         # True to use LLM for extraction
    extractor_model: str = "gpt-4o-mini"   # Model for LLM extractor

    # LLM judge model (can differ from main model)
    judge_model: Optional[str] = None      # Defaults to main model

    # Retriever
    retriever_top_k: int = 10
    retriever_use_embeddings: bool = False

    # Fail-safe behavior when uncertain
    fail_safe: str = "repair"              # "allow" | "repair" | "block"

    # Logging
    log_all_turns: bool = False
    emit_python_logs: bool = True

    # Block message template
    block_message: str = (
        "I'm unable to provide that response as it conflicts with established "
        "policies or prior information in this conversation. "
        "Please contact support if you need further assistance."
    )

    # Escalation
    escalate_on_policy: bool = True


# ---------------------------------------------------------------------------
# Block message builder
# ---------------------------------------------------------------------------

def _build_block_response(
    session_id: str,
    turn: int,
    events: List[ContraEvent],
    config: FirewallConfig,
    model_latency: float,
    firewall_latency: float,
    raw_content: str,
) -> FirewallResponse:
    return FirewallResponse(
        content=config.block_message,
        was_repaired=False,
        was_blocked=True,
        raw_content=raw_content,
        contra_events=events,
        action=ActionDecision.BLOCK,
        model_latency_ms=model_latency,
        firewall_latency_ms=firewall_latency,
        session_id=session_id,
        turn=turn,
    )


# ---------------------------------------------------------------------------
# Main middleware
# ---------------------------------------------------------------------------

class FirewallMiddleware:
    """
    The Contradiction Firewall middleware.

    Wraps OpenAI and Anthropic LLM calls with:
      - Claim-level contradiction detection
      - Multi-judge scoring (rule, NLI, LLM, numeric)
      - Self-repair loop
      - Hard constraint ledger enforcement
      - Structured audit logging

    Parameters
    ----------
    provider : str
        "openai" or "anthropic"
    model : str
        LLM model name (e.g. "gpt-4o", "claude-3-5-sonnet-20241022")
    api_key : str, optional
        API key (defaults to env var OPENAI_API_KEY / ANTHROPIC_API_KEY)
    ledger : ConstraintLedger, optional
        Hard constraint store. Violations always surfaced regardless of confidence.
    config : FirewallConfig, optional
        Full configuration. Defaults are production-safe.
    logger : FirewallLogger, optional
        Custom logging configuration. Defaults to in-memory + Python logger.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        ledger: Optional[ConstraintLedger] = None,
        config: Optional[FirewallConfig] = None,
        logger: Optional[FirewallLogger] = None,
        # Convenience kwargs (forwarded to FirewallConfig)
        block_threshold: float = 0.85,
        repair_threshold: float = 0.55,
        memory_window: int = 10,
        max_repair_attempts: int = 2,
        use_nli: bool = True,
        use_llm_judge: bool = True,
    ) -> None:
        self.provider = provider
        self.model = model

        # Build config
        if config is None:
            config = FirewallConfig(
                block_threshold=block_threshold,
                repair_threshold=repair_threshold,
                memory_window=memory_window,
                max_repair_attempts=max_repair_attempts,
                use_nli=use_nli,
                use_llm_judge=use_llm_judge,
            )
        self.config = config

        # Initialize LLM client
        self._client = self._init_client(provider, api_key)

        # Initialize components
        self._ledger = ledger or ConstraintLedger()
        self._memory = ConversationMemory(window=config.memory_window)
        self._memory.session_id = str(uuid.uuid4())

        self._extractor = ClaimExtractor(
            use_llm=config.use_llm_extractor,
            llm_client=self._client if config.use_llm_extractor else None,
            model=config.extractor_model,
            provider=provider,
        )

        self._retriever = CandidateRetriever(
            top_k=config.retriever_top_k,
            use_embeddings=config.retriever_use_embeddings,
        )

        risk_config = RiskConfig(
            log_threshold=config.log_threshold,
            repair_threshold=config.repair_threshold,
            block_threshold=config.block_threshold,
            escalate_on_policy=config.escalate_on_policy,
            fail_safe=config.fail_safe,
        )
        self._risk_engine = RiskEngine(config=risk_config)

        self._repair_layer = RepairLayer(
            llm_client=self._client,
            model=model,
            provider=provider,
            max_attempts=config.max_repair_attempts,
        )

        self._logger = logger or FirewallLogger(
            sinks=[InMemoryLogSink()],
            emit_to_python_logger=config.emit_python_logs,
            log_all_turns=config.log_all_turns,
        )

        # Lazy-initialize detectors
        self._detectors = self._init_detectors()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        **llm_kwargs: Any,
    ) -> FirewallResponse:
        """
        Main entry point. Wraps a single LLM call with the full firewall pipeline.

        Parameters
        ----------
        messages : list
            OpenAI-style message list: [{"role": "user", "content": "..."}]
        system : str, optional
            System prompt. Also extracted for claim comparison.
        **llm_kwargs
            Additional kwargs forwarded to the LLM API call (temperature, max_tokens, etc.)

        Returns
        -------
        FirewallResponse
            Contains the (possibly repaired or blocked) content + full audit report.
        """
        fw_start = time.monotonic()
        self._memory._turn_counter += 1
        turn = self._memory._turn_counter

        # 1. Call LLM
        model_start = time.monotonic()
        raw_content = self._call_llm(messages, system=system, **llm_kwargs)
        model_latency = (time.monotonic() - model_start) * 1000

        # 2. Extract claims from response
        response_claims = self._extractor.extract(
            raw_content, source="response", turn=turn
        )

        # 3. Extract system prompt claims (once per conversation, if changed)
        system_claims: List[Claim] = []
        if system:
            system_claims = self._extractor.extract(system, source="system_prompt", turn=0)

        # 4. Get all comparison targets: memory + ledger + system prompt
        ledger_claims = self._ledger.all_claims()
        memory_claims = self._memory.all_claims(role_filter="assistant")
        all_prior_claims = ledger_claims + memory_claims + system_claims

        # 5. Run detection pipeline for each response claim
        all_events: List[ContraEvent] = []
        for candidate_claim in response_claims:
            relevant_priors = self._retriever.retrieve(
                candidate=candidate_claim,
                memory_claims=memory_claims + system_claims,
                ledger_claims=ledger_claims,
            )
            if not relevant_priors:
                continue

            for prior_claim in relevant_priors:
                det_results = self._run_detectors(candidate_claim, prior_claim)
                event = self._risk_engine.adjudicate(
                    candidate=candidate_claim,
                    conflicting=prior_claim,
                    detector_results=det_results,
                    turn=turn,
                    session_id=self._memory.session_id,
                )
                # Only keep events above log threshold
                if event.combined_confidence >= self.config.log_threshold:
                    all_events.append(event)

        # 6. Determine overall action (worst-case across all events)
        overall_action = self._worst_case_action(all_events)

        # 7. Act
        firewall_latency = (time.monotonic() - fw_start) * 1000
        response = self._execute_action(
            action=overall_action,
            raw_content=raw_content,
            events=all_events,
            messages=messages,
            system=system,
            turn=turn,
            model_latency=model_latency,
            firewall_latency=firewall_latency,
            llm_kwargs=llm_kwargs,
        )

        # 8. Store assistant turn in memory (store repaired content)
        final_claims = (
            self._extractor.extract(response.content, source="response", turn=turn)
            if response.was_repaired
            else response_claims
        )
        self._memory.add_turn("assistant", response.content, claims=final_claims)

        # 9. Log
        self._logger.log_response(response)

        return response

    @property
    def session_id(self) -> str:
        return self._memory.session_id

    def reset_session(self) -> None:
        """Start a new session (clear memory, new session ID)."""
        self._memory.clear()
        self._memory.session_id = str(uuid.uuid4())

    @property
    def memory(self) -> ConversationMemory:
        return self._memory

    @property
    def logger(self) -> FirewallLogger:
        return self._logger

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llm(
        self, messages: List[Dict], system: Optional[str], **kwargs: Any
    ) -> str:
        """Call the configured LLM provider."""
        if self.provider == "openai":
            full_messages = []
            if system:
                full_messages.append({"role": "system", "content": system})
            full_messages.extend(messages)
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                **kwargs,
            )
            return resp.choices[0].message.content or ""

        elif self.provider == "anthropic":
            create_kwargs = dict(
                model=self.model,
                max_tokens=kwargs.pop("max_tokens", 4096),
                messages=messages,
                **kwargs,
            )
            if system:
                create_kwargs["system"] = system
            resp = self._client.messages.create(**create_kwargs)
            return resp.content[0].text

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _run_detectors(self, candidate: Claim, prior: Claim):
        """Run all enabled detectors on a (candidate, prior) pair."""
        from .models import DetectorResult
        results = []

        for detector in self._detectors:
            try:
                result = detector.check_pair(candidate, prior)
                results.append(result)
            except Exception:
                pass  # Never let detector failure break the response

        return results

    def _worst_case_action(self, events: List[ContraEvent]) -> ActionDecision:
        """Return the most severe action across all contradiction events."""
        severity_order = [
            ActionDecision.ALLOW,
            ActionDecision.LOG_ONLY,
            ActionDecision.REPAIR,
            ActionDecision.BLOCK,
            ActionDecision.ESCALATE,
        ]
        if not events:
            return ActionDecision.ALLOW
        worst = ActionDecision.ALLOW
        for event in events:
            if severity_order.index(event.action) > severity_order.index(worst):
                worst = event.action
        return worst

    def _execute_action(
        self,
        action: ActionDecision,
        raw_content: str,
        events: List[ContraEvent],
        messages: List[Dict],
        system: Optional[str],
        turn: int,
        model_latency: float,
        firewall_latency: float,
        llm_kwargs: Dict,
    ) -> FirewallResponse:
        """Execute the decided action and return a FirewallResponse."""

        if action in (ActionDecision.ALLOW, ActionDecision.LOG_ONLY):
            return FirewallResponse(
                content=raw_content,
                was_repaired=False,
                was_blocked=False,
                raw_content=raw_content,
                contra_events=events,
                action=action,
                model_latency_ms=model_latency,
                firewall_latency_ms=firewall_latency,
                session_id=self._memory.session_id,
                turn=turn,
            )

        if action == ActionDecision.REPAIR:
            fw_start = time.monotonic()
            repaired, success, repair_explanation = self._repair_layer.repair(
                original_response=raw_content,
                events=events,
                recheck_fn=None,  # Re-check would require another full pipeline run
            )
            firewall_latency += (time.monotonic() - fw_start) * 1000

            # Update events with repair info
            for event in events:
                event.repair_attempted = True
                event.repair_succeeded = success
                event.repair_explanation = repair_explanation

            if success:
                return FirewallResponse(
                    content=repaired,
                    was_repaired=True,
                    was_blocked=False,
                    raw_content=raw_content,
                    contra_events=events,
                    action=ActionDecision.REPAIR,
                    model_latency_ms=model_latency,
                    firewall_latency_ms=firewall_latency,
                    session_id=self._memory.session_id,
                    turn=turn,
                )
            else:
                # Repair failed -- block
                return _build_block_response(
                    self._memory.session_id, turn, events, self.config,
                    model_latency, firewall_latency, raw_content
                )

        if action in (ActionDecision.BLOCK, ActionDecision.ESCALATE):
            return _build_block_response(
                self._memory.session_id, turn, events, self.config,
                model_latency, firewall_latency, raw_content
            )

        # Fallback
        return FirewallResponse(
            content=raw_content,
            contra_events=events,
            action=ActionDecision.ALLOW,
            session_id=self._memory.session_id,
            turn=turn,
        )

    def _init_client(self, provider: str, api_key: Optional[str]) -> Any:
        """Initialize the LLM API client."""
        if provider == "openai":
            try:
                from openai import OpenAI
                return OpenAI(api_key=api_key) if api_key else OpenAI()
            except ImportError:
                raise ImportError("OpenAI provider requires: pip install openai")
        elif provider == "anthropic":
            try:
                from anthropic import Anthropic
                return Anthropic(api_key=api_key) if api_key else Anthropic()
            except ImportError:
                raise ImportError("Anthropic provider requires: pip install anthropic")
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'.")

    def _init_detectors(self) -> list:
        """Initialize enabled detectors."""
        from .detectors.rule_based import RuleBasedDetector
        from .detectors.numeric import NumericTemporalDetector

        detectors: list = [RuleBasedDetector(), NumericTemporalDetector()]

        if self.config.use_nli:
            try:
                from .detectors.nli import NLIDetector
                detectors.append(
                    NLIDetector(
                        model_name=self.config.nli_model,
                        device=self.config.nli_device,
                    )
                )
            except ImportError:
                pass  # NLI optional -- requires sentence-transformers

        if self.config.use_llm_judge:
            from .detectors.llm_judge import LLMJudge
            judge_model = self.config.judge_model or self.model
            detectors.append(
                LLMJudge(
                    llm_client=self._client,
                    model=judge_model,
                    provider=self.provider,
                )
            )

        return detectors
