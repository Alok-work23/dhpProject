import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class FinBERTSentiment:
    def __init__(self):
        model_name = "ProsusAI/finbert"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.id2label = self.model.config.id2label
        self.confidence_threshold = 0.6

    def analyze(self, text):
        if not text or not isinstance(text, str):
            return {"label": "neutral", "score": 0.0}
        try:
            inputs = self.tokenizer(text[:512], return_tensors="pt", truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                max_prob, pred_idx = torch.max(probs, dim=1)
                label = self.id2label[pred_idx.item()].lower()
                return {
                    "label": label if max_prob.item() >= self.confidence_threshold else "neutral",
                    "score": round(max_prob.item(), 4)
                }
        except Exception as e:
            print(f"Sentiment analysis failed: {e}")
            return {"label": "neutral", "score": 0.0}

    def pipeline(self, texts):
        results = []
        for text in texts:
            results.append(self.analyze(text))
        return results
