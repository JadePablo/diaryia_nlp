import tweetnlp

class ModelSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize_models()
        return cls._instance

    def initialize_models(self):
        self.topic_model = tweetnlp.TopicClassification()
        self.emo_model = tweetnlp.Emotion()
        self.ner_model = tweetnlp.NER()