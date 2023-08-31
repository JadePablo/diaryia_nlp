import numpy as np
from model_handler import ModelSingleton

model_instance = ModelSingleton()

emo_model = model_instance.emo_model
topic_model = model_instance.topic_model
ner_model = model_instance.ner_model

def entity_mapTo_emotion(sentence_arr):
    entity_segmentation = {}
    last_entity = "unknown"
    entity_segmentation[last_entity] = ""

    for sentence in sentence_arr:
        ner_result = ner_model.ner(sentence)

        if len(ner_result) > 0:
            entities = np.array([detection["entity"].replace(" ", "").lower() for detection in ner_result])
            unique_entities, entity_counts = np.unique(entities, return_counts=True)

            for entity, count in zip(unique_entities, entity_counts):
                entity_segmentation[entity] = entity_segmentation.get(entity, "") + sentence * count
                last_entity = entity
        else:
            entity_segmentation[last_entity] += sentence

    return entity_segmentation

def topic_mapTo_emotion(sentence_arr):
    topic_segmentation = {}

    for sentence in sentence_arr:
        topic_result = topic_model.topic(sentence)
        topics = topic_result['label']
        unique_topics, topic_counts = np.unique(topics, return_counts=True)

        for topic, count in zip(unique_topics, topic_counts):
            topic_segmentation[topic] = topic_segmentation.get(topic, "") + sentence * count

    return topic_segmentation

def apply_logic_to_sentence_arr(sentence_arr, logic_function):
    result_segmentation = {}

    for sentence in sentence_arr:
        logic_result = logic_function(sentence)
        logic_keys = np.array(list(logic_result.keys()))
        unique_keys, key_counts = np.unique(logic_keys, return_counts=True)

        for key, count in zip(unique_keys, key_counts):
            result_segmentation[key] = result_segmentation.get(key, "") + logic_result[key] * count

    return result_segmentation

def entity_mapTo_emotion_numpy(sentence_arr):
    def entity_logic(sentence):
        return entity_mapTo_emotion([sentence])
    
    return apply_logic_to_sentence_arr(sentence_arr, entity_logic)

def topic_mapTo_emotion_numpy(sentence_arr):
    def topic_logic(sentence):
        return topic_mapTo_emotion([sentence])
    
    return apply_logic_to_sentence_arr(sentence_arr, topic_logic)
