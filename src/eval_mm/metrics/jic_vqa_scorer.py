from .scorer import Scorer, AggregateOutput


class JICVQAScorer(Scorer):
    @staticmethod
    def score(refs: list[str], preds: list[str], **kwargs) -> list[int]:
        scores = [int(ref in pred) for ref, pred in zip(refs, preds)]
        return scores

    @staticmethod
    def aggregate(scores: list[int], **kwargs) -> AggregateOutput:
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
            result["average"] = sum(domain_averages) / len(domain_averages)
        else:
            result["average"] = 0.0
        output = AggregateOutput(result["average"], result)
        return output


def test_jic_vqa_test():
    refs = ["私は猫です。"]
    preds = ["私は猫です。"]
    scores = JICVQAScorer.score(refs, preds, docs={"domain": ["test"]})
    assert scores == [1]
    output = JICVQAScorer.aggregate(scores, docs={"domain": ["test"]})
    assert output.overall_score == 1.0
    assert output.details == {"test": 1.0, "average": 1.0}
