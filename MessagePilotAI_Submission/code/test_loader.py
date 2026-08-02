import unittest
import os
from loader import DataLoader
from models import MessageContext, Message, User, Group, Business
from evidence_retriever import EvidenceRetriever, EvidenceDetail
from llm_reasoner import LLMReasoner

class TestDataLayerRetrieverAndLLM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.dataset_dir = os.path.join(base_dir, "dataset")
        cls.loader = DataLoader(cls.dataset_dir)
        cls.loader.load_all()
        cls.retriever = EvidenceRetriever(cls.loader)
        cls.reasoner = LLMReasoner()

    def test_messages_loaded(self):
        self.assertGreater(len(self.loader.messages), 0, "Messages should be loaded")
        first_msg = self.loader.messages[0]
        self.assertIsInstance(first_msg, Message)

    def test_evidence_retrieval_enhanced(self):
        sample_msg = self.loader.messages[0]
        ctx = self.loader.build_message_context(sample_msg)
        details = self.retriever.retrieve_evidence_details(ctx, top_k=3)
        self.assertIsInstance(details, list)

    def test_llm_reasoner_fallback_and_schema(self):
        sample_msg = self.loader.messages[0]
        ctx = self.loader.build_message_context(sample_msg)
        details = self.retriever.retrieve_evidence_details(ctx, top_k=3)
        
        # Test reasoner (will use baseline fallback if no API key is present in env)
        result = self.reasoner.reason(
            context=ctx,
            features={"is_dnd": False, "is_opted_out": False},
            baseline_result={"action": "notify", "message_type": "personal", "reason": "Baseline recommendation", "confidence": 0.8, "evidence_message_ids": ["msg_001"]},
            evidence_items=details
        )
        
        self.assertIn("action", result)
        self.assertIn("message_type", result)
        self.assertIn("reason", result)
        self.assertIn("confidence", result)
        self.assertIn("evidence_message_ids", result)
        self.assertIn(result["action"], ["notify", "digest", "mute"])

if __name__ == '__main__':
    unittest.main()
