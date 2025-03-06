from .scorer import Scorer


class JICVQAScorer(Scorer):
    @staticmethod
    def score(refs: list[str], preds: list[str], **kwargs) -> list[int]:
        scores = [int(ref in pred) for ref, pred in zip(refs, preds)]
        return scores

    @staticmethod
    def aggregate(scores: list[int], **kwargs) -> float:
        docs = kwargs["docs"]
        domain_scores = {}

        # Accumulate scores for each domain and overall
        for i, domain in enumerate(docs["domain"]):
            if domain not in domain_scores:
                domain_scores[domain] = {"total_score": 0, "count": 0}
            domain_scores[domain]["total_score"] += scores[i]
            domain_scores[domain]["count"] += 1

        # Calculate average for each domain
        result = {}
        domain_averages = []
        for domain, values in domain_scores.items():
            average = values["total_score"] / values["count"]
            result[domain] = average
            domain_averages.append(average)

        # Calculate the average of all domain averages
        if domain_averages:
            result['average'] = sum(domain_averages) / len(domain_averages)
        else:
            result['average'] = 0.0

        return result
