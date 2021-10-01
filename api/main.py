from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List
from similarity import FaissSimilarity
import uvicorn
import uuid
import psycopg2
import datetime
import pickle
import sys
import psycopg2.extras
psycopg2.extras.register_uuid()

sys.path.append('../')
from config import config


class SimilarityInputTraining(BaseModel):
    task_name: str
    train_data: List


class SimilarityInputInference(BaseModel):
    model_id: str
    inference_data: List


app = FastAPI(
    title="Similarity Search",
    description="""Customizable Similarity Search Based on FAISS"""
)


def training_job(model_id, task_name, input_data):
    bytes_training_data = pickle.dumps(input_data)
    bytes_training_reference = pickle.dumps(
        {k: v for k, v in enumerate(input_data)})
    training_created_at = datetime.datetime.now()
    training_updated_at = training_created_at
    model_id = str(model_id)

    autosim = FaissSimilarity()
    index = autosim.create_index(input_data)
    bytes_index = pickle.dumps(index)

    params = config(filename='../database.ini', section='postgresql')
    conn = psycopg2.connect(**params
                            )
    cur = conn.cursor()
    cur.execute("INSERT INTO models (id, index, task_name, reference_data, reference_dict, created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s)", (
        model_id, bytes_index, task_name, bytes_training_data, bytes_training_reference, training_created_at, training_updated_at))
    conn.commit()
    cur.close()
    conn.close()


def inference_job(model_id, input_data):
    bytes_inference_data = pickle.dumps(input_data)

    params = config(filename='../database.ini', section='postgresql')
    conn = psycopg2.connect(**params)

    cur = conn.cursor()
    cur.execute(
        "SELECT index, reference_dict FROM models WHERE id = %s;", (str(model_id),))
    result = cur.fetchone()
    cur.close()
    conn.close()

    autosim = FaissSimilarity()
    autosim.load_index_from_bytes(result[0], result[1])
    result = autosim.get_similar_items(input_data, k=3)

    print(result)


@app.post('/')
def ping():
    return{'msg': 'acknowledged'}


@app.post('/similarity/train')
def train_similarity(similarity_input_training: SimilarityInputTraining, background_tasks: BackgroundTasks):
    data = similarity_input_training.dict()
    model_id = uuid.uuid4()
    background_tasks.add_task(training_job, model_id, data['task_name'], data['train_data'])
    return {"generated_id": model_id}


@app.post('/similarity/inference')
def inference_similarity(similarity_input_inference: SimilarityInputInference, background_tasks: BackgroundTasks):
    data = similarity_input_inference.dict()
    model_id = data['model_id']
    result_id = uuid.uuid4()
    background_tasks.add_task(inference_job, model_id, data['inference_data'])
    return {"generated_id": result_id}
