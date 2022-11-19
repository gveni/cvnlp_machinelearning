from transformers import pipeline

# Sentiment analysis
#classifier = pipeline("sentiment-analysis")
#print(classifier(["This code is about exploring various HuggingFace Use-Cases", "I am not that strong at NLP", "I am better at Computer vision"]))

# Zero-shot learning
#classifier = pipeline("zero-shot-classification")
#print(classifier("I am learning transformers via HuggingFace",
#                  candidate_labels=["education", "politics", "sports", "business"]))

# Text generation
#generator = pipeline("text-generation", model="distilgpt2")
#print(generator("Today, I am trying to learn",
#                num_return_sequences=3, max_length=20))

# Mask filling
#unmasker = pipeline("fill-mask")
#print(unmasker("Today, I am trying to learn about <mask> models", top_k = 3))

# NER
#ner = pipeline("ner", grouped_entities=True)
#print(ner("My name is Gopal Veni and I work for Ancestry in Lehi, Utah"))

# Question answering
#question_answer = pipeline("question-answering")
#print(question_answer(question="Where do I work",
#                context="I work for Ancestry in Lehi, UT"))

# Text summarization
#text_summarizer = pipeline("summarization")
#print(text_summarizer(
#    """
#    When I saw Elon Musk the first time and spoke with him, he was very happy. He was stoked, which is an extreme form of happy.
#    When I saw him during a dinner in LA a few months later, he was very unhappy, and felt treated very unfairly by the media (about the disproportionate attention given to battery fires in Model S's).
#    So the answer is: Sometimes Elon Musk is happy and sometimes Elon Musk is not happy.
#    He's a human being, so he feels different things at different times.
#    I don't agree that it's a stupid question.
#    Some people aren't sad or happy. Some people take Prozac and other drugs just so they don't feel a lot of emotions or ups and downs. If you don't know anyone like this, you might be more isolated from humanity than you imagine.
#    """
#))

# Translation
fr2en_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
print(fr2en_translator("Je suis etudiant"))