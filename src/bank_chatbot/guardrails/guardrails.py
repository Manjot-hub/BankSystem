"""
NeMo Guardrails Integration

Wraps NeMo Guardrails with banking-specific policies and PII detection.
Supports Groq and OpenAI for LLM-based rails evaluation.
"""
import os
import re
from typing import Any, Optional
from pathlib import Path

from src.bank_chatbot.config.settings import get_settings


class BankingGuardrails:
    """Banking-specific guardrails using NeMo Guardrails and regex heuristics."""

    def __init__(self, config_dir: Optional[str] = None):
        self.settings = get_settings()
        self.config_dir = Path(config_dir or self.settings.GUARDRAILS_CONFIG_DIR)
        self.rails = None

        if self.config_dir.exists():
            try:
                from nemoguardrails import LLMRails, RailsConfig
                llm_instance = self._get_llm()
                if llm_instance:
                    self.config = RailsConfig.from_path(str(self.config_dir))
                    self.rails = LLMRails(self.config, llm=llm_instance)
                else:
                    print("NeMo Guardrails skipped: No valid LLM credentials found. Using Python-based guardrails.")
            except Exception as e:
                print(f"Warning: Could not initialize NeMo Guardrails: {e}")
        else:
            print("NeMo Guardrails config dir not found. Using Python-based guardrails.")

    def _get_llm(self):
        """Get LLM instance (Groq primary, OpenAI fallback) for guardrails."""
        groq_api_key = getattr(self.settings, "GROQ_API_KEY", None) or os.getenv("GROQ_API_KEY")
        if groq_api_key:
            try:
                from langchain_groq import ChatGroq
                return ChatGroq(
                    model_name=getattr(self.settings, "LLM_MODEL_PRIMARY", "llama-3.3-70b-versatile"),
                    groq_api_key=groq_api_key,
                    temperature=getattr(self.settings, "LLM_TEMPERATURE", 0.1),
                )
            except ImportError:
                pass

        openai_api_key = getattr(self.settings, "OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=getattr(self.settings, "LLM_MODEL_PRIMARY", "gpt-4o-mini"),
                temperature=getattr(self.settings, "LLM_TEMPERATURE", 0.1),
                openai_api_key=openai_api_key,
            )
        return None

    def process_message(self, message: str, context: Optional[dict] = None) -> dict[str, Any]:
        """Process a message through guardrails."""
        result = {
            "allowed": True,
            "message": message,
            "flags": [],
            "blocked": False,
            "requires_human": False,
        }

        # 1. PII Detection
        pii_flags = self._detect_pii(message)
        if pii_flags:
            result["flags"].extend(pii_flags)
            result["allowed"] = False
            result["blocked"] = True

        # 2. Prompt Injection Detection
        injection_flags = self._detect_injection(message)
        if injection_flags:
            result["flags"].extend(injection_flags)
            result["allowed"] = False
            result["blocked"] = True

        # 3. Financial Advice Detection
        advice_flags = self._detect_financial_advice(message)
        if advice_flags:
            result["flags"].extend(advice_flags)
            result["allowed"] = False
            result["blocked"] = True

        # 4. NeMo Guardrails evaluation (if enabled)
        if self.rails and result["allowed"]:
            try:
                guardrail_result = self.rails.generate(
                    messages=[{"role": "user", "content": message}]
                )
                result["rails_response"] = guardrail_result
            except Exception as e:
                result["flags"].append(f"guardrails_error:{str(e)[:100]}")
                result["requires_human"] = True

        return result

    def _detect_pii(self, message: str) -> list[str]:
        """Detect personally identifiable information."""
        flags = []
        if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", message):
            flags.append("pii_email")
        if re.search(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b", message):
            flags.append("pii_phone")
        if re.search(r"\b\d{3}-\d{2}-\d{4}\b", message):
            flags.append("pii_ssn")
        if re.search(r"\b\d{8,17}\b", message):
            flags.append("pii_account_number")
        if re.search(r"\b(?:\d{4}[-\s]?){3}\d{1,4}\b", message):
            flags.append("pii_credit_card")
        return flags

    def _detect_injection(self, message: str) -> list[str]:
        """Detect prompt injection attempts."""
        flags = []
        lower = message.lower()
        injection_patterns = [
            "ignore previous instructions", "disregard all previous", "system prompt",
            "developer instructions", "bypass security", "jailbreak", "ignore all rules",
            "show me your prompt", "reveal your instructions", "you are now",
            "act as system", "sudo", "rm -rf", "drop table", "select * from",
        ]
        for pattern in injection_patterns:
            if pattern in lower:
                flags.append(f"injection:{pattern}")
        return flags

    def _detect_financial_advice(self, message: str) -> list[str]:
        """Detect requests for financial, investment, tax, or legal advice."""
        flags = []
        lower = message.lower()
        advice_patterns = [
            "investment advice", "guaranteed return", "guaranteed returns", "risk-free",
            "risk free", "what stock should i buy", "should i invest", "best investment",
            "tax advice", "legal advice", "should i take a loan", "loan advice",
        ]
        for pattern in advice_patterns:
            if pattern in lower:
                flags.append(f"financial_advice:{pattern}")
        return flags

    def validate_response(self, response: str, context: Optional[dict] = None) -> dict[str, Any]:
        """Validate a generated response before sending to user."""
        result = {
            "allowed": True,
            "response": response,
            "flags": [],
            "blocked": False,
        }
        forbidden_patterns = [
            r"guaranteed return", r"risk[- ]free", r"investment advice",
            r"tax advice", r"legal advice", r"definitely", r"always",
            r"never worry", r"no risk",
        ]
        for pattern in forbidden_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                result["flags"].append(f"forbidden:{pattern}")
                result["allowed"] = False
                result["blocked"] = True

        pii_flags = self._detect_pii(response)
        if pii_flags:
            result["flags"].extend(pii_flags)
            result["allowed"] = False
            result["blocked"] = True

        return result

    def get_policy_summary(self) -> dict[str, Any]:
        """Return a summary of active guardrail policies."""
        return {
            "pii_detection": True,
            "prompt_injection_detection": True,
            "forbidden_content_detection": True,
            "nemo_rails_enabled": self.rails is not None,
            "config_dir": str(self.config_dir),
        }