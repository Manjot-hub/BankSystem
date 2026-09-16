"""
Evaluation Harness for RAG Pipeline

Implements comprehensive evaluation with golden set, metrics, and CI gates.
"""
import json
from pathlib import Path
from typing import Any
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np

from src.bank_chatbot.config.settings import get_settings
from src.bank_chatbot.rag.pipeline import RAGPipeline


@dataclass
class EvaluationResult:
    """Single evaluation result."""
    query: str
    expected_answer: str
    generated_answer: str
    retrieved_contexts: list[str]
    faithfulness_score: float
    relevancy_score: float
    precision_score: float
    recall_score: float
    latency_ms: float
    passed: bool


@dataclass
class EvaluationSummary:
    """Aggregated evaluation summary."""
    total_queries: int
    passed: int
    failed: int
    avg_faithfulness: float
    avg_relevancy: float
    avg_precision: float
    avg_recall: float
    avg_latency_ms: float
    timestamp: str
    threshold_faithfulness: float
    threshold_relevancy: float
    threshold_precision: float


class EvaluationHarness:
    """Evaluation harness for RAG pipeline."""

    def __init__(self):
        self.settings = get_settings()
        self.pipeline = RAGPipeline()
        self.golden_set_path = Path(self.settings.EVAL_GOLDEN_SET_PATH)
        self.golden_set_path.parent.mkdir(parents=True, exist_ok=True)

    def create_golden_set(self) -> list[dict[str, Any]]:
        """Create golden set of Q&A pairs for evaluation."""
        golden_set = [
            {
                "query": "What is the funds availability policy for check deposits?",
                "expected_answer": "Cash deposits and the first $225 of non-cash deposits are available on the first business day. Local checks take 2 business days, non-local checks take 5 business days.",
                "context_keywords": ["funds availability", "check", "deposit", "business day", "hold"]
            },
            {
                "query": "How do I report a lost or stolen debit card?",
                "expected_answer": "Report immediately via the mobile app (Cards > Select Card > Report Lost/Stolen) or call 1-800-XXX-XXXX. Card is instantly deactivated. Replacement mailed in 5-7 business days. Zero liability for unauthorized transactions.",
                "context_keywords": ["lost", "stolen", "debit card", "report", "deactivated", "replacement", "zero liability"]
            },
            {
                "query": "What are the wire transfer fees?",
                "expected_answer": "Domestic outgoing: $25, Domestic incoming: $15. International outgoing: $45, International incoming: $20. Cutoff: 4 PM ET domestic, 3 PM ET international.",
                "context_keywords": ["wire", "transfer", "fee", "domestic", "international", "cutoff"]
            },
            {
                "query": "What is the overdraft fee?",
                "expected_answer": "Overdraft fee is $35 per item, maximum 4 per day ($140). Continuous overdraft fee: $10/day after 5 days. Overdraft protection transfer: $12.50.",
                "context_keywords": ["overdraft", "fee", "$35", "per item", "continuous", "protection"]
            },
            {
                "query": "What are the mobile check deposit limits?",
                "expected_answer": "Daily limit: $5,000. Monthly limit: $15,000. Per check limit: $5,000. Funds available next business day for first $225. Endorse with 'For Mobile Deposit Only'.",
                "context_keywords": ["mobile", "check", "deposit", "limit", "daily", "monthly", "endorse"]
            },
            {
                "query": "How do I set up direct deposit?",
                "expected_answer": "Get account and routing numbers from app (Account Details). Provide to employer payroll department. Or download pre-filled form from app > Account > Direct Deposit. First deposit takes 1-2 pay cycles.",
                "context_keywords": ["direct", "deposit", "routing", "account number", "payroll", "form"]
            },
            {
                "query": "What is the difference between ACH and wire transfer?",
                "expected_answer": "ACH: Usually free, 1-3 business days, lower limits, reversible within 60 days. Wire: $25 domestic/$45 international, same day domestic, higher limits, very difficult to reverse.",
                "context_keywords": ["ach", "wire", "transfer", "cost", "speed", "reversible", "limits"]
            },
            {
                "query": "How do I reset my online banking password?",
                "expected_answer": "Click 'Forgot Password' on login page. Enter username and registered email/phone. Choose verification method (SMS, email, authenticator). Enter code and create new 12+ character password. Visit branch with ID if no access to verification methods.",
                "context_keywords": ["password", "reset", "forgot", "verification", "mfa", "branch"]
            },
            {
                "query": "What is the dispute resolution process for unauthorized transactions?",
                "expected_answer": "Report within 60 days of statement. Zero liability if reported within 2 business days. Up to $50 liability within 60 days. Unlimited liability after 60 days. Bank provisionally credits within 10 business days. Investigation completes within 45 days (90 for POS/foreign).",
                "context_keywords": ["dispute", "unauthorized", "transaction", "reg e", "liability", "provisional credit", "investigation"]
            },
            {
                "query": "What are the monthly maintenance fees for checking accounts?",
                "expected_answer": "Basic Checking: $12 (waived with $1,500 avg balance or $500 direct deposit). Premium Checking: $25 (waived with $15,000 combined balances). Student Checking: $0 (age 17-24). Senior Checking: $0 (age 65+).",
                "context_keywords": ["monthly", "maintenance", "fee", "checking", "waived", "balance", "direct deposit", "student", "senior"]
            },
            {
                "query": "What are the ATM fees for out-of-network and international withdrawals?",
                "expected_answer": "In-network: $0. Out-of-network US: $3.00. International: $5.00 + 3% currency conversion fee.",
                "context_keywords": ["atm", "fee", "out-of-network", "international", "currency conversion"]
            },
        ]

        # Save golden set
        with open(self.golden_set_path, "w") as f:
            json.dump(golden_set, f, indent=2)

        return golden_set

    def load_golden_set(self) -> list[dict[str, Any]]:
        """Load golden set from file."""
        if not self.golden_set_path.exists():
            return self.create_golden_set()

        with open(self.golden_set_path) as f:
            return json.load(f)

    def evaluate_single(self, query: str, expected_answer: str) -> EvaluationResult:
        """Evaluate a single query."""
        import time

        start_time = time.time()

        # Run pipeline
        result = self.pipeline.invoke(query)

        latency_ms = (time.time() - start_time) * 1000

        # Prepare retrieved contexts from pipeline result
        retrieved_docs = result.get("retrieved_docs", [])
        retrieved_contexts = [doc.page_content for doc in retrieved_docs]

        # Compute metrics using our simplified implementations
        faithfulness_score = self._compute_faithfulness(
            result["answer"], retrieved_contexts
        )
        relevancy_score = self._compute_relevancy(query, result["answer"])
        precision_score = self._compute_precision(query, retrieved_contexts)
        recall_score = self._compute_recall(query, expected_answer, retrieved_contexts)

        # Determine pass/fail
        passed = (
            faithfulness_score >= self.settings.EVAL_THRESHOLD_FAITHFULNESS and
            relevancy_score >= self.settings.EVAL_THRESHOLD_RELEVANCE and
            precision_score >= self.settings.EVAL_THRESHOLD_PRECISION
        )

        return EvaluationResult(
            query=query,
            expected_answer=expected_answer,
            generated_answer=result["answer"],
            retrieved_contexts=retrieved_contexts,
            faithfulness_score=faithfulness_score,
            relevancy_score=relevancy_score,
            precision_score=precision_score,
            recall_score=recall_score,
            latency_ms=latency_ms,
            passed=passed,
        )

    def _compute_faithfulness(self, answer: str, contexts: list[str]) -> float:
        """Compute faithfulness score (simplified heuristic)."""
        if not contexts:
            return 0.0

        # Simple heuristic: check if answer references context
        answer_lower = answer.lower()
        context_text = " ".join(contexts).lower()

        # Check for citation patterns
        has_citations = "[source" in answer_lower or "source " in answer_lower

        # Check overlap with context
        answer_words = set(answer_lower.split())
        context_words = set(context_text.split())
        overlap = len(answer_words & context_words) / max(len(answer_words), 1)

        # Base score from overlap + citation bonus
        score = overlap * 0.7 + (0.3 if has_citations else 0)
        return min(score, 1.0)

    def _compute_relevancy(self, query: str, answer: str) -> float:
        """Compute answer relevancy score (simplified)."""
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())

        # Remove stop words
        stop_words = {"what", "is", "the", "a", "an", "how", "do", "i", "for", "to", "of", "and", "or", "in", "on", "with", "my", "your", "can", "you", "me", "tell", "about"}
        query_words = query_words - stop_words
        answer_words = answer_words - stop_words

        if not query_words:
            return 0.5

        overlap = len(query_words & answer_words) / len(query_words)
        return min(overlap * 1.2, 1.0)

    def _compute_precision(self, query: str, contexts: list[str]) -> float:
        """Compute context precision (simplified)."""
        if not contexts:
            return 0.0

        query_words = set(query.lower().split())
        stop_words = {"what", "is", "the", "a", "an", "how", "do", "i", "for", "to", "of", "and", "or", "in", "on", "with"}
        query_words = query_words - stop_words

        if not query_words:
            return 0.5

        # Check how many contexts contain query terms
        relevant = 0
        for ctx in contexts:
            ctx_words = set(ctx.lower().split())
            if query_words & ctx_words:
                relevant += 1

        return relevant / len(contexts)

    def _compute_recall(self, query: str, expected: str, contexts: list[str]) -> float:
        """Compute context recall (simplified)."""
        expected_words = set(expected.lower().split())
        stop_words = {"what", "is", "the", "a", "an", "how", "do", "i", "for", "to", "of", "and", "or", "in", "on", "with", "you", "your", "can", "me", "tell", "about"}
        expected_words = expected_words - stop_words

        if not expected_words:
            return 0.5

        # Check how many expected terms appear in contexts
        context_text = " ".join(contexts).lower()
        found = sum(1 for w in expected_words if w in context_text)

        return found / len(expected_words)

    def run_evaluation(self) -> EvaluationSummary:
        """Run full evaluation on golden set."""
        golden_set = self.load_golden_set()

        print(f"Running evaluation on {len(golden_set)} queries...")
        results = []

        for item in golden_set:
            print(f"  Evaluating: {item['query'][:50]}...")
            result = self.evaluate_single(item["query"], item["expected_answer"])
            results.append(result)

        # Aggregate
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed

        summary = EvaluationSummary(
            total_queries=len(results),
            passed=passed,
            failed=failed,
            avg_faithfulness=np.mean([r.faithfulness_score for r in results]),
            avg_relevancy=np.mean([r.relevancy_score for r in results]),
            avg_precision=np.mean([r.precision_score for r in results]),
            avg_recall=np.mean([r.recall_score for r in results]),
            avg_latency_ms=np.mean([r.latency_ms for r in results]),
            timestamp=datetime.now().isoformat(),
            threshold_faithfulness=self.settings.EVAL_THRESHOLD_FAITHFULNESS,
            threshold_relevancy=self.settings.EVAL_THRESHOLD_RELEVANCE,
            threshold_precision=self.settings.EVAL_THRESHOLD_PRECISION,
        )

        return summary, results

    def print_summary(self, summary: EvaluationSummary, results: list[EvaluationResult]):
        """Print evaluation summary."""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Queries: {summary.total_queries}")
        print(f"Passed: {summary.passed}")
        print(f"Failed: {summary.failed}")
        print(f"Pass Rate: {summary.passed/summary.total_queries*100:.1f}%")
        print(f"\nMetrics:")
        print(f"  Faithfulness: {summary.avg_faithfulness:.3f} (threshold: {summary.threshold_faithfulness})")
        print(f"  Relevancy:    {summary.avg_relevancy:.3f} (threshold: {summary.threshold_relevancy})")
        print(f"  Precision:    {summary.avg_precision:.3f} (threshold: {summary.threshold_precision})")
        print(f"  Recall:       {summary.avg_recall:.3f}")
        print(f"  Avg Latency:  {summary.avg_latency_ms:.1f}ms")
        print(f"\nTimestamp: {summary.timestamp}")

        print("\nFailed Queries:")
        for r in results:
            if not r.passed:
                print(f"  - {r.query[:60]}... (F:{r.faithfulness_score:.2f}, R:{r.relevancy_score:.2f}, P:{r.precision_score:.2f})")

    def save_results(self, summary: EvaluationSummary, results: list[EvaluationResult]):
        """Save evaluation results to file."""
        output = {
            "summary": asdict(summary),
            "results": [asdict(r) for r in results],
        }

        output_path = Path("data/eval/results_latest.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\nResults saved to {output_path}")

    def check_gates(self, summary: EvaluationSummary) -> bool:
        """Check if evaluation passes CI gates."""
        gates = [
            ("Faithfulness", summary.avg_faithfulness >= summary.threshold_faithfulness),
            ("Relevancy", summary.avg_relevancy >= summary.threshold_relevancy),
            ("Precision", summary.avg_precision >= summary.threshold_precision),
            ("Pass Rate", summary.passed / summary.total_queries >= 0.5),  # 50% for PoC
        ]

        print("\nCI Gates:")
        all_passed = True
        for name, passed in gates:
            status = "PASS" if passed else "FAIL"
            print(f"  {name}: {status}")
            if not passed:
                all_passed = False

        return all_passed


def main():
    """Run evaluation harness."""
    harness = EvaluationHarness()

    # Create golden set if not exists
    if not harness.golden_set_path.exists():
        print("Creating golden set...")
        harness.create_golden_set()

    # Run evaluation
    summary, results = harness.run_evaluation()

    # Print summary
    harness.print_summary(summary, results)

    # Save results
    harness.save_results(summary, results)

    # Check gates
    gates_passed = harness.check_gates(summary)

    if not gates_passed:
        print("\n❌ CI Gates FAILED - Evaluation did not meet thresholds")
        sys.exit(1)
    else:
        print("\n✅ CI Gates PASSED - All thresholds met")
        sys.exit(0)


if __name__ == "__main__":
    import sys
    main()