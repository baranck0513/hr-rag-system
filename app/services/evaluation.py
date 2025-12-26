"""
Evaluation Service

Metrics for measuring RAG system quality:
- Recall@k: Are we finding all relevant documents?
- Precision@k: What percentage of returned documents are relevant?
- MRR (Mean Reciprocal Rank): Is the best answer ranked first?

These metrics require a "ground truth" dataset - a set of queries
with known relevant documents to compare against.
"""

from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvaluationQuery:
    """
    A single evaluation query with ground truth.
    
    Attributes:
        query: The question being asked
        relevant_ids: IDs of documents/chunks that are relevant to this query
        retrieved_ids: IDs of documents/chunks that were actually retrieved
    """
    query: str
    relevant_ids: list[str]
    retrieved_ids: list[str] = field(default_factory=list)
    
    @property
    def has_results(self) -> bool:
        """Check if any results were retrieved."""
        return len(self.retrieved_ids) > 0
    
    @property
    def has_relevant(self) -> bool:
        """Check if there are known relevant documents."""
        return len(self.relevant_ids) > 0


@dataclass
class EvaluationResult:
    """
    Results of evaluating a single query.
    
    Attributes:
        query: The original query
        recall: Recall score (0-1)
        precision: Precision score (0-1)
        reciprocal_rank: Reciprocal rank (0-1)
        relevant_found: Number of relevant documents found
        relevant_total: Total number of relevant documents
        retrieved_total: Total number of retrieved documents
    """
    query: str
    recall: float
    precision: float
    reciprocal_rank: float
    relevant_found: int
    relevant_total: int
    retrieved_total: int


@dataclass
class AggregateResults:
    """
    Aggregated evaluation results across multiple queries.
    
    Attributes:
        mean_recall: Average recall across all queries
        mean_precision: Average precision across all queries
        mrr: Mean Reciprocal Rank across all queries
        total_queries: Number of queries evaluated
        individual_results: Results for each query
    """
    mean_recall: float
    mean_precision: float
    mrr: float
    total_queries: int
    individual_results: list[EvaluationResult] = field(default_factory=list)
    
    def summary(self) -> str:
        """Return a formatted summary of results."""
        return (
            f"Evaluation Results ({self.total_queries} queries)\n"
            f"{'=' * 40}\n"
            f"Mean Recall@k:    {self.mean_recall:.3f}\n"
            f"Mean Precision@k: {self.mean_precision:.3f}\n"
            f"MRR:              {self.mrr:.3f}\n"
        )


class EvaluationService:
    """
    Service for evaluating RAG system quality.
    
    Usage:
        evaluator = EvaluationService()
        
        # Evaluate a single query
        result = evaluator.evaluate_query(
            query="How much leave do I get?",
            relevant_ids=["chunk_1", "chunk_5"],
            retrieved_ids=["chunk_3", "chunk_1", "chunk_7"]
        )
        
        # Evaluate multiple queries
        results = evaluator.evaluate_batch(queries)
    """
    
    def __init__(self, k: int = 5):
        """
        Initialise the evaluation service.
        
        Args:
            k: The number of top results to consider (for @k metrics)
        """
        self.k = k
    
    def recall_at_k(
        self,
        relevant_ids: list[str],
        retrieved_ids: list[str],
        k: Optional[int] = None
    ) -> float:
        """
        Calculate Recall@k.
        
        Recall@k = (relevant documents in top-k) / (total relevant documents)
        
        Measures: Of all relevant documents, how many did we find?
        
        Args:
            relevant_ids: IDs of documents that are relevant
            retrieved_ids: IDs of documents that were retrieved (in order)
            k: Number of top results to consider (defaults to self.k)
            
        Returns:
            Recall score between 0 and 1
        """
        if not relevant_ids:
            # No relevant documents - recall is undefined, return 1.0
            logger.warning("No relevant documents provided for recall calculation")
            return 1.0
        
        k = k or self.k
        top_k_retrieved = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        relevant_found = len(top_k_retrieved & relevant_set)
        
        return relevant_found / len(relevant_set)
    
    def precision_at_k(
        self,
        relevant_ids: list[str],
        retrieved_ids: list[str],
        k: Optional[int] = None
    ) -> float:
        """
        Calculate Precision@k.
        
        Precision@k = (relevant documents in top-k) / k
        
        Measures: Of the documents we returned, how many are relevant?
        
        Args:
            relevant_ids: IDs of documents that are relevant
            retrieved_ids: IDs of documents that were retrieved (in order)
            k: Number of top results to consider (defaults to self.k)
            
        Returns:
            Precision score between 0 and 1
        """
        k = k or self.k
        top_k_retrieved = retrieved_ids[:k]
        
        if not top_k_retrieved:
            return 0.0
        
        relevant_set = set(relevant_ids)
        relevant_found = sum(1 for doc_id in top_k_retrieved if doc_id in relevant_set)
        
        return relevant_found / len(top_k_retrieved)
    
    def reciprocal_rank(
        self,
        relevant_ids: list[str],
        retrieved_ids: list[str]
    ) -> float:
        """
        Calculate Reciprocal Rank.
        
        RR = 1 / (rank of first relevant document)
        
        Measures: How high is the first relevant result ranked?
        
        Args:
            relevant_ids: IDs of documents that are relevant
            retrieved_ids: IDs of documents that were retrieved (in order)
            
        Returns:
            Reciprocal rank between 0 and 1 (0 if no relevant found)
        """
        if not relevant_ids or not retrieved_ids:
            return 0.0
        
        relevant_set = set(relevant_ids)
        
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_set:
                return 1.0 / rank
        
        # No relevant document found
        return 0.0
    
    def evaluate_query(
        self,
        query: str,
        relevant_ids: list[str],
        retrieved_ids: list[str],
        k: Optional[int] = None
    ) -> EvaluationResult:
        """
        Evaluate a single query.
        
        Args:
            query: The query text
            relevant_ids: IDs of relevant documents
            retrieved_ids: IDs of retrieved documents (in order)
            k: Number of top results to consider
            
        Returns:
            EvaluationResult with all metrics
        """
        k = k or self.k
        
        recall = self.recall_at_k(relevant_ids, retrieved_ids, k)
        precision = self.precision_at_k(relevant_ids, retrieved_ids, k)
        rr = self.reciprocal_rank(relevant_ids, retrieved_ids)
        
        # Count relevant found
        top_k_retrieved = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        relevant_found = len(top_k_retrieved & relevant_set)
        
        return EvaluationResult(
            query=query,
            recall=recall,
            precision=precision,
            reciprocal_rank=rr,
            relevant_found=relevant_found,
            relevant_total=len(relevant_ids),
            retrieved_total=len(retrieved_ids)
        )
    
    def evaluate_batch(
        self,
        queries: list[EvaluationQuery],
        k: Optional[int] = None
    ) -> AggregateResults:
        """
        Evaluate multiple queries and aggregate results.
        
        Args:
            queries: List of EvaluationQuery objects
            k: Number of top results to consider
            
        Returns:
            AggregateResults with mean metrics
        """
        if not queries:
            return AggregateResults(
                mean_recall=0.0,
                mean_precision=0.0,
                mrr=0.0,
                total_queries=0
            )
        
        k = k or self.k
        results = []
        
        for eq in queries:
            result = self.evaluate_query(
                query=eq.query,
                relevant_ids=eq.relevant_ids,
                retrieved_ids=eq.retrieved_ids,
                k=k
            )
            results.append(result)
        
        # Calculate means
        mean_recall = sum(r.recall for r in results) / len(results)
        mean_precision = sum(r.precision for r in results) / len(results)
        mrr = sum(r.reciprocal_rank for r in results) / len(results)
        
        return AggregateResults(
            mean_recall=mean_recall,
            mean_precision=mean_precision,
            mrr=mrr,
            total_queries=len(queries),
            individual_results=results
        )


class RetrievalEvaluator:
    """
    Higher-level evaluator that works with a Retriever.
    
    Runs queries through the retriever and evaluates results
    against ground truth.
    
    Usage:
        evaluator = RetrievalEvaluator(retriever)
        results = evaluator.evaluate(test_dataset)
    """
    
    def __init__(self, retriever, eval_service: Optional[EvaluationService] = None):
        """
        Initialise the evaluator.
        
        Args:
            retriever: The retriever to evaluate
            eval_service: Evaluation service (creates default if not provided)
        """
        self.retriever = retriever
        self.eval_service = eval_service or EvaluationService()
    
    def evaluate(
        self,
        test_queries: list[dict],
        k: int = 5,
        id_field: str = "document_id"
    ) -> AggregateResults:
        """
        Evaluate the retriever on a test dataset.
        
        Args:
            test_queries: List of dicts with 'query' and 'relevant_ids' keys
            k: Number of results to retrieve
            id_field: Field name to extract IDs from results
            
        Returns:
            AggregateResults with evaluation metrics
        """
        evaluation_queries = []
        
        for test in test_queries:
            query = test["query"]
            relevant_ids = test["relevant_ids"]
            
            # Run the retriever
            result = self.retriever.retrieve(query=query, top_k=k)
            
            # Extract retrieved IDs
            retrieved_ids = [
                r.metadata.get(id_field, str(i))
                for i, r in enumerate(result.results)
            ]
            
            evaluation_queries.append(EvaluationQuery(
                query=query,
                relevant_ids=relevant_ids,
                retrieved_ids=retrieved_ids
            ))
        
        return self.eval_service.evaluate_batch(evaluation_queries, k=k)