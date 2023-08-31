#get the emotions of a text
#get the topics of a text
#get the entities of a text

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
                
        if (len(ner_result) > 0) :
            for detection in ner_result:

                entity = detection["entity"].replace(" ", "").lower()

                if entity in entity_segmentation:
                    entity_segmentation[entity] += sentence
                    last_entity = entity
                else:
                    entity_segmentation[entity] = sentence
                    last_entity = entity


        else:
            entity_segmentation[last_entity] += sentence 
    
    for entity in entity_segmentation:
        entity_segmentation[entity] = emo_model.predict(entity_segmentation[entity])

    return entity_segmentation
        


def topic_mapTo_emotion(sentence_arr):
    
    topic_segmentation = {}


    for sentence in sentence_arr:
        topic_result = topic_model.topic(sentence)
        
        for topic in topic_result['label']:
            if topic in topic_segmentation:
                topic_segmentation[topic] += sentence
            else:
                topic_segmentation[topic] = sentence

    for topic in topic_segmentation:
        topic_segmentation[topic] = emo_model.predict(topic_segmentation[topic])

    return topic_segmentation