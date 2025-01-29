from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import QuestionGenerator
from haystack.pipelines import QuestionGenerationPipeline
import model

tokenizer, model = model.load_model()

def doc_store_init(context):
    document_store = InMemoryDocumentStore()
    docs = [{"content": context}]
    document_store.write_documents(docs)
    return document_store

def generate_questions(question_pipeline, docs):
    generated_questions = []
    for document in docs:
        result = question_pipeline.run(documents=[document])
        questions = result['generated_questions'][0]['questions']
        generated_questions.extend(questions)
    return generated_questions

def question_init(context):
    question_generator = QuestionGenerator()
    question_generation_pipeline = QuestionGenerationPipeline(question_generator)
    docs = doc_store_init(context).get_all_documents()
    questions = generate_questions(question_generation_pipeline, docs)
    return questions

def generate_answer(context, question):
    input_text = f"Answer the following question: {question} in a long paragraph based on the context: {context}"
    input_tokens = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=1024)
    outputs = model.generate(input_ids=input_tokens['input_ids'], attention_mask=input_tokens['attention_mask'], num_beams=5, min_length=70, max_length=2000, early_stopping=True)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def generate_questions_and_answers(context):
    questions = question_init(context)
    answers = []
    for question in questions:
        answer = generate_answer(context, question)
        answers.append({"question": question, "answer": answer})
    return answers