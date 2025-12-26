"""
Tests for the evaluation service.
"""

import pytest
from app.services.evaluation import (
    EvaluationQuery,
    EvaluationResult,
    AggregateResults,
    EvaluationService,
    RetrievalEvaluator,
)


class TestEvaluationQuery:
    """Tests for the EvaluationQuery dataclass."""
    
    def test_query_creation(self):
        """Should create an evaluation query."""
        eq = EvaluationQuery(
            query="How much leave?",
            relevant_ids=["doc1", "doc2"],
            retrieved_ids=["doc3", "doc1"]
        )
        
        assert eq.query == "How much leave?"
        assert eq.relevant_ids == ["doc1", "doc2"]
        assert eq.retrieved_ids == ["doc3", "doc1"]
    
    def test_has_results_true(self):
        """Should return True when results exist."""
        eq = EvaluationQuery(
            query="test",
            relevant_ids=["doc1"],
            retrieved_ids=["doc1", "doc2"]
        )
        
        assert eq.has_results is True
    
    def test_has_results_false(self):
        """Should return False when no results."""
        eq = EvaluationQuery(
            query="test",
            relevant_ids=["doc1"],
            retrieved_ids=[]
        )
        
        assert eq.has_results is False
    
    def test_has_relevant_true(self):
        """Should return True when relevant docs exist."""
        eq = EvaluationQuery(
            query="test",
            relevant_ids=["doc1"],
            retrieved_ids=[]
        )
        
        assert eq.has_relevant is True
    
    def test_has_relevant_false(self):
        """Should return False when no relevant docs."""
        eq = EvaluationQuery(
            query="test",
            relevant_ids=[],
            retrieved_ids=["doc1"]
        )
        
        assert eq.has_relevant is False


class TestEvaluationService:
    """Tests for the EvaluationService class."""
    
    @pytest.fixture
    def evaluator(self):
        """Create an evaluation service with k=5."""
        return EvaluationService(k=5)
    
    # --- Recall@k Tests ---
    
    def test_recall_perfect(self, evaluator):
        """Should return 1.0 when all relevant docs are retrieved."""
        relevant = ["doc1", "doc2"]
        retrieved = ["doc1", "doc2", "doc3"]
        
        recall = evaluator.recall_at_k(relevant, retrieved)
        
        assert recall == 1.0
    
    def test_recall_partial(self, evaluator):
        """Should return correct ratio for partial recall."""
        relevant = ["doc1", "doc2", "doc3", "doc4"]
        retrieved = ["doc1", "doc2", "doc5"]  # Found 2 of 4
        
        recall = evaluator.recall_at_k(relevant, retrieved)
        
        assert recall == 0.5
    
    def test_recall_none_found(self, evaluator):
        """Should return 0 when no relevant docs found."""
        relevant = ["doc1", "doc2"]
        retrieved = ["doc3", "doc4", "doc5"]
        
        recall = evaluator.recall_at_k(relevant, retrieved)
        
        assert recall == 0.0
    
    def test_recall_respects_k(self, evaluator):
        """Should only consider top-k results."""
        relevant = ["doc5"]  # Only doc5 is relevant
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"]
        
        # With k=5, doc5 is at position 5 (included)
        recall_k5 = evaluator.recall_at_k(relevant, retrieved, k=5)
        assert recall_k5 == 1.0
        
        # With k=4, doc5 is not in top-4
        recall_k4 = evaluator.recall_at_k(relevant, retrieved, k=4)
        assert recall_k4 == 0.0
    
    def test_recall_empty_retrieved(self, evaluator):
        """Should return 0 when nothing retrieved."""
        relevant = ["doc1"]
        retrieved = []
        
        recall = evaluator.recall_at_k(relevant, retrieved)
        
        assert recall == 0.0
    
    # --- Precision@k Tests ---
    
    def test_precision_perfect(self, evaluator):
        """Should return 1.0 when all retrieved are relevant."""
        relevant = ["doc1", "doc2", "doc3"]
        retrieved = ["doc1", "doc2"]
        
        precision = evaluator.precision_at_k(relevant, retrieved, k=2)
        
        assert precision == 1.0
    
    def test_precision_partial(self, evaluator):
        """Should return correct ratio for partial precision."""
        relevant = ["doc1", "doc2"]
        retrieved = ["doc1", "doc3", "doc4", "doc5"]  # 1 relevant in top-4
        
        precision = evaluator.precision_at_k(relevant, retrieved, k=4)
        
        assert precision == 0.25
    
    def test_precision_none_relevant(self, evaluator):
        """Should return 0 when no retrieved are relevant."""
        relevant = ["doc1", "doc2"]
        retrieved = ["doc3", "doc4", "doc5"]
        
        precision = evaluator.precision_at_k(relevant, retrieved, k=3)
        
        assert precision == 0.0
    
    def test_precision_empty_retrieved(self, evaluator):
        """Should return 0 when nothing retrieved."""
        relevant = ["doc1"]
        retrieved = []
        
        precision = evaluator.precision_at_k(relevant, retrieved)
        
        assert precision == 0.0
    
    # --- Reciprocal Rank Tests ---
    
    def test_rr_first_position(self, evaluator):
        """Should return 1.0 when relevant doc is first."""
        relevant = ["doc1"]
        retrieved = ["doc1", "doc2", "doc3"]
        
        rr = evaluator.reciprocal_rank(relevant, retrieved)
        
        assert rr == 1.0
    
    def test_rr_second_position(self, evaluator):
        """Should return 0.5 when relevant doc is second."""
        relevant = ["doc1"]
        retrieved = ["doc2", "doc1", "doc3"]
        
        rr = evaluator.reciprocal_rank(relevant, retrieved)
        
        assert rr == 0.5
    
    def test_rr_third_position(self, evaluator):
        """Should return 0.333 when relevant doc is third."""
        relevant = ["doc1"]
        retrieved = ["doc2", "doc3", "doc1"]
        
        rr = evaluator.reciprocal_rank(relevant, retrieved)
        
        assert abs(rr - 0.333) < 0.01
    
    def test_rr_not_found(self, evaluator):
        """Should return 0 when no relevant doc found."""
        relevant = ["doc1"]
        retrieved = ["doc2", "doc3", "doc4"]
        
        rr = evaluator.reciprocal_rank(relevant, retrieved)
        
        assert rr == 0.0
    
    def test_rr_multiple_relevant(self, evaluator):
        """Should use rank of FIRST relevant doc."""
        relevant = ["doc3", "doc5"]  # Two relevant docs
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        
        rr = evaluator.reciprocal_rank(relevant, retrieved)
        
        # First relevant (doc3) is at position 3
        assert abs(rr - 0.333) < 0.01
    
    def test_rr_empty_inputs(self, evaluator):
        """Should return 0 for empty inputs."""
        assert evaluator.reciprocal_rank([], ["doc1"]) == 0.0
        assert evaluator.reciprocal_rank(["doc1"], []) == 0.0
    
    # --- evaluate_query Tests ---
    
    def test_evaluate_query_complete(self, evaluator):
        """Should return all metrics for a query."""
        result = evaluator.evaluate_query(
            query="How much leave?",
            relevant_ids=["doc1", "doc2"],
            retrieved_ids=["doc3", "doc1", "doc4", "doc2", "doc5"],
            k=5
        )
        
        assert result.query == "How much leave?"
        assert result.recall == 1.0  # Found both relevant
        assert result.precision == 0.4  # 2 relevant in 5 retrieved
        assert result.reciprocal_rank == 0.5  # First relevant at position 2
        assert result.relevant_found == 2
        assert result.relevant_total == 2
    
    # --- evaluate_batch Tests ---
    
    def test_evaluate_batch(self, evaluator):
        """Should aggregate results across queries."""
        queries = [
            EvaluationQuery(
                query="Query 1",
                relevant_ids=["doc1"],
                retrieved_ids=["doc1", "doc2"]  # Perfect RR=1.0
            ),
            EvaluationQuery(
                query="Query 2",
                relevant_ids=["doc1"],
                retrieved_ids=["doc2", "doc1"]  # RR=0.5
            ),
        ]
        
        results = evaluator.evaluate_batch(queries)
        
        assert results.total_queries == 2
        assert results.mrr == 0.75  # (1.0 + 0.5) / 2
        assert len(results.individual_results) == 2
    
    def test_evaluate_batch_empty(self, evaluator):
        """Should handle empty query list."""
        results = evaluator.evaluate_batch([])
        
        assert results.total_queries == 0
        assert results.mrr == 0.0


class TestAggregateResults:
    """Tests for the AggregateResults dataclass."""
    
    def test_summary_format(self):
        """Should return formatted summary string."""
        results = AggregateResults(
            mean_recall=0.8,
            mean_precision=0.6,
            mrr=0.75,
            total_queries=10
        )
        
        summary = results.summary()
        
        assert "10 queries" in summary
        assert "0.800" in summary  # Recall
        assert "0.600" in summary  # Precision
        assert "0.750" in summary  # MRR


class TestEvaluationRealWorldScenarios:
    """Integration tests with realistic scenarios."""
    
    @pytest.fixture
    def evaluator(self):
        return EvaluationService(k=5)
    
    def test_scenario_perfect_retrieval(self, evaluator):
        """Scenario: System returns exactly the right documents."""
        result = evaluator.evaluate_query(
            query="What is the leave policy?",
            relevant_ids=["leave_doc_1", "leave_doc_2"],
            retrieved_ids=["leave_doc_1", "leave_doc_2", "other_doc"]
        )
        
        assert result.recall == 1.0
        assert result.reciprocal_rank == 1.0
    
    def test_scenario_good_but_not_first(self, evaluator):
        """Scenario: Correct answer found but not at top."""
        result = evaluator.evaluate_query(
            query="How much annual leave?",
            relevant_ids=["annual_leave_chunk"],
            retrieved_ids=[
                "sick_leave_chunk",      # Rank 1 - wrong
                "annual_leave_chunk",    # Rank 2 - correct!
                "holiday_chunk"          # Rank 3 - wrong
            ]
        )
        
        assert result.recall == 1.0  # Found it
        assert result.reciprocal_rank == 0.5  # But at position 2
    
    def test_scenario_missed_relevant(self, evaluator):
        """Scenario: System missed some relevant documents."""
        result = evaluator.evaluate_query(
            query="Salary information",
            relevant_ids=["salary_bands", "bonus_policy", "pay_review"],
            retrieved_ids=[
                "salary_bands",     # Found
                "holiday_policy",   # Wrong
                "expenses_policy"   # Wrong
            ]
        )
        
        assert abs(result.recall - 0.333) < 0.01  # Found 1 of 3
        assert result.reciprocal_rank == 1.0  # First result was relevant
    
    def test_scenario_complete_miss(self, evaluator):
        """Scenario: System returned completely wrong results."""
        result = evaluator.evaluate_query(
            query="Maternity leave policy",
            relevant_ids=["maternity_doc"],
            retrieved_ids=[
                "annual_leave_doc",
                "sick_leave_doc",
                "expenses_doc"
            ]
        )
        
        assert result.recall == 0.0
        assert result.precision == 0.0
        assert result.reciprocal_rank == 0.0
    
    def test_scenario_mrr_calculation(self, evaluator):
        """Scenario: Calculate MRR across multiple queries (from earlier example)."""
        queries = [
            EvaluationQuery(
                query="Query 1",
                relevant_ids=["doc1"],
                retrieved_ids=["doc1", "doc2", "doc3"]  # RR = 1.0
            ),
            EvaluationQuery(
                query="Query 2",
                relevant_ids=["doc1"],
                retrieved_ids=["doc2", "doc1", "doc3"]  # RR = 0.5
            ),
            EvaluationQuery(
                query="Query 3",
                relevant_ids=["doc1"],
                retrieved_ids=["doc2", "doc3", "doc4", "doc1"]  # RR = 0.25
            ),
        ]
        
        results = evaluator.evaluate_batch(queries)
        
        # MRR = (1.0 + 0.5 + 0.25) / 3 = 0.583
        assert abs(results.mrr - 0.583) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])