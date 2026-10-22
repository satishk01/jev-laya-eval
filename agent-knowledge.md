# AI Engineering Basics

## What AI Engineering Is

AI engineering is the practice of building reliable software products on top of pretrained machine learning models, especially large language models. An AI engineer usually does not train a model from scratch, because that requires huge datasets, specialised hardware and a dedicated research team. Instead, an AI engineer connects existing models to data, tools and user interfaces to solve a concrete problem.

The discipline sits between traditional software engineering and machine learning research. It borrows testing, version control, monitoring and deployment practices from software engineering. It borrows ideas such as evaluation datasets, metrics and error analysis from machine learning.

A typical AI engineering task might be building a chatbot that answers questions about a company's internal documents. Another common task is extracting structured fields, such as names and dates, from unstructured text like emails or invoices. A third example is building an assistant that can call software tools to complete multi-step tasks on behalf of a user.

The core building blocks of AI engineering include prompts, models, retrieval systems, tools and evaluation pipelines. Prompts tell the model what to do, and retrieval systems supply the model with relevant information it did not learn during training. Tools let the model take actions or fetch live data, while evaluation pipelines measure whether the whole system behaves well.

A key difference from traditional software is that model outputs are probabilistic rather than fully deterministic. The same input can produce different outputs, and small wording changes in a prompt can change results noticeably. Because of this, AI engineers rely heavily on systematic testing with many examples rather than a handful of manual checks.

Good AI engineering is iterative, starting with a simple baseline and improving it based on measured results. Engineers typically try prompting first, then add retrieval or tools, and only consider fine-tuning when simpler approaches fall short. Throughout this process, they track quality, latency and cost together, because improving one often affects the others.

Responsible AI engineering also includes thinking about safety, privacy and the impact of mistakes on users. A system that gives medical, legal or financial information needs stronger safeguards than one that suggests recipe ideas. Ultimately, the goal of AI engineering is useful, trustworthy behaviour in production, not just an impressive demo.

## Large Language Models

A large language model, often shortened to LLM, is a neural network trained to predict the next piece of text given the text that came before it. Most modern LLMs are based on the transformer architecture, which uses a mechanism called attention to relate every token in the input to every other token. Attention lets the model weigh which earlier words matter most when predicting the next one.

LLMs are first trained in a stage called pretraining, where they read enormous amounts of text from books, websites and code. During pretraining, the model learns grammar, facts, reasoning patterns and writing styles purely by predicting the next token over and over. The knowledge learned this way is stored implicitly in the model's parameters, which are the numerical weights of the network.

The number of parameters is one rough indicator of a model's size and capacity. Larger models generally know more and reason better, but they are also slower and more expensive to run. Smaller models can be faster and cheaper, and they are often good enough for narrow, well-defined tasks.

After pretraining, models usually go through further training to make them follow instructions and behave helpfully. This stage often includes supervised fine-tuning on example conversations and reinforcement learning from human feedback, commonly called RLHF. The result is an instruction-tuned or chat model that responds to requests instead of simply continuing text.

An LLM has a training cutoff, which is the point in time after which it has no knowledge from its training data. The model does not automatically know about events, documents or data created after that cutoff. This is one major reason engineers connect models to retrieval systems and tools that provide fresh information.

LLMs generate output one token at a time in a process called autoregressive generation. At each step, the model produces a probability distribution over its vocabulary, picks one token, appends it to the text, and repeats. Generation stops when the model emits a special end token or reaches a configured maximum length.

It is important to remember that an LLM does not look up facts in a database when it answers. It produces text that is statistically likely given its training, which is usually correct for common facts but can be wrong for rare or specific ones. Understanding this limitation helps engineers design systems that check and ground the model's answers.

## Tokens and Tokenization

A token is the basic unit of text that a language model reads and writes. Tokens are often pieces of words rather than whole words, so a long or unusual word may be split into several tokens. Common short words, on the other hand, are frequently represented by a single token.

Tokenization is the process of converting raw text into a sequence of tokens. Each token is mapped to an integer ID from the model's fixed vocabulary, and the model actually processes these IDs rather than letters. When the model produces output, its token IDs are converted back into readable text in a step called detokenization.

Many tokenizers use an algorithm called byte pair encoding, often abbreviated as BPE. Byte pair encoding starts with individual characters or bytes and repeatedly merges the most frequent adjacent pairs into new tokens. This produces a vocabulary that handles common words efficiently while still being able to represent any text.

As a rough rule of thumb for English text, one token corresponds to about four characters or about three quarters of a word. This ratio varies by language, and many non-English languages use more tokens for the same meaning. Code, numbers and unusual symbols can also consume more tokens than plain English prose.

Tokens matter because model limits and pricing are usually measured in tokens rather than words or characters. The maximum amount of text a model can handle at once is expressed as a token count. Hosted model providers typically charge separately for input tokens and output tokens.

Different model families use different tokenizers, so the same sentence can produce different token counts on different models. Engineers should use the tokenizer that matches their chosen model when they need accurate counts. Many providers offer a tokenizer library or a token-counting endpoint for this purpose.

Tokenization also explains some surprising model weaknesses. Because a model sees tokens rather than individual letters, tasks like counting letters in a word or reversing a string can be harder than expected. Arithmetic on long numbers can also be unreliable because digits may be grouped into tokens inconsistently.

## Context Windows

The context window is the maximum number of tokens a model can consider at one time. It includes everything the model sees in a single request: the system prompt, the conversation history, any retrieved documents, tool results and the model's own output. If the total exceeds the context window, the request will fail or older content must be removed.

Context windows vary widely between models, ranging from a few thousand tokens to hundreds of thousands or more. A larger context window lets an application include longer documents or longer conversations in a single request. However, a larger window does not guarantee that the model will use all of that information equally well.

Research and practical experience show that models sometimes pay less attention to information buried in the middle of a very long context. This effect is often called the lost in the middle problem. Placing the most important instructions and facts near the beginning or end of the prompt can help.

Filling the context window also has practical costs. More input tokens mean higher cost per request and usually higher latency before the first output token appears. For this reason, engineers try to include only the information that is actually relevant to the current task.

The context window is separate from the model's long-term knowledge stored in its parameters. Anything placed in the context window is available only for that single request unless the application sends it again. A model does not remember earlier conversations unless the application includes them in the new request.

The maximum output length is often configured separately from the context window. Many APIs have a parameter, often called max tokens, that limits how many tokens the model may generate in its reply. If this limit is too low, the answer may be cut off mid-sentence.

Engineers manage context with techniques such as truncation, summarisation and retrieval. Truncation drops the oldest or least relevant content, while summarisation compresses earlier content into a shorter form. Retrieval avoids loading everything at once by fetching only the pieces most relevant to the current question.

## Prompts and Prompt Engineering

A prompt is the input text given to a language model to tell it what to do. Prompt engineering is the practice of designing and refining prompts so that the model produces useful, accurate and consistent outputs. It is usually the first and cheapest lever an AI engineer has for improving a system.

Clear and specific instructions are the foundation of a good prompt. Instead of asking a model to summarise a document, a better prompt states the audience, the desired length and the format of the summary. Vague prompts tend to produce vague or inconsistent answers.

Few-shot prompting means including a small number of worked examples of inputs and desired outputs in the prompt. Examples show the model the expected format and style more reliably than descriptions alone. Zero-shot prompting, by contrast, gives instructions without any examples.

Chain-of-thought prompting asks the model to reason step by step before giving a final answer. This technique often improves accuracy on problems involving arithmetic, logic or multi-step reasoning. Some applications hide the reasoning from the end user and show only the final answer.

Asking for a specific output format reduces the need for fragile parsing later. A prompt might request a numbered list, a single word, or a JSON object with particular fields. Stating what to do when information is missing, such as replying that the answer is unknown, also improves reliability.

Prompt engineering is an empirical process. Engineers should change one thing at a time and compare results on a fixed set of test inputs. Prompts should be stored in version control so that changes can be tracked, reviewed and rolled back like any other code.

A prompt template is a prompt with placeholders that are filled in at runtime with user input or retrieved data. Templates make prompts reusable and consistent across many requests. Care is needed when inserting untrusted user text into templates, because it can contain instructions that conflict with the intended behaviour.

## System Prompts and Roles

Most chat-based model APIs organise a conversation as a list of messages, each with a role. The common roles are system, user and assistant, and some APIs add a tool role for tool results. The role tells the model who wrote each message and how much authority it should carry.

The system prompt is a special message that sets the overall behaviour, persona and rules for the assistant. It is usually written by the application developer rather than by the end user. A system prompt might define the assistant's tone, the topics it should cover, and the format of its replies.

Models are generally trained to give system instructions higher priority than user instructions. This priority is helpful but not absolute, so a determined user may still persuade the model to break system rules. For that reason, system prompts should not be treated as a complete security boundary.

A good system prompt is specific about what the assistant should do and what it should avoid. It can describe the target audience, such as beginners, and instruct the assistant to explain jargon. It can also tell the assistant to say when it does not know an answer instead of guessing.

System prompts are a good place to state rules about using provided context. For example, a system prompt in a retrieval application might tell the model to answer only from the supplied documents. It might also require the model to cite which document each claim came from.

Keeping secrets in a system prompt is a bad practice. Users can often trick a model into revealing its system prompt through clever questions. Any credentials, private keys or sensitive business data should be kept out of prompts entirely.

Long system prompts consume tokens on every request, which increases cost and latency. Some providers support prompt caching, which can reduce the cost of repeatedly sending the same prefix. Even with caching, it is worth keeping system prompts focused and free of unnecessary text.

## Sampling: Temperature and Top-p

When a model generates text, it produces a probability for every token in its vocabulary at each step. Sampling is the process of choosing which token to output from that probability distribution. Sampling parameters control how random or predictable the resulting text will be.

Temperature is the most common sampling parameter. A low temperature makes the model favour the most probable tokens, producing more focused and repeatable output. A high temperature flattens the distribution, making less likely tokens more probable and the output more varied and creative.

A temperature of zero, or very close to zero, approximates greedy decoding, where the single most likely token is chosen each time. Greedy decoding is useful for tasks that need consistency, such as classification or data extraction. Even at temperature zero, outputs from hosted APIs are not always perfectly identical because of hardware and batching effects.

Top-p sampling, also called nucleus sampling, restricts choices to the smallest set of tokens whose combined probability reaches a threshold p. For example, with a top-p of 0.9, the model samples only from the tokens that together make up ninety percent of the probability. This cuts off the long tail of unlikely tokens while keeping some variety.

Providers usually recommend adjusting either temperature or top-p, but not both at the same time. Changing both makes it harder to understand which setting caused a change in behaviour. For most applications, adjusting temperature alone is sufficient.

Low temperatures suit factual question answering, code generation and structured extraction. Higher temperatures suit brainstorming, creative writing and generating diverse alternatives. There is no universally best value, so engineers should test settings on their own tasks.

Sampling settings interact with evaluation. If a system uses a non-zero temperature, the same test input may pass once and fail the next time. Running each test case several times and looking at the pass rate gives a more honest picture of reliability.

## Structured Output and Constrained Decoding

Structured output means getting a model to return data in a predictable, machine-readable format rather than free-form prose. The most common structured format is JSON, which programs can parse directly into objects. Structured output is essential when model responses feed into other software.

The simplest way to get structured output is to ask for it in the prompt and show an example of the expected format. This approach works reasonably well but is not guaranteed, because the model may add extra commentary or produce invalid syntax. Applications that rely only on prompting should always validate and handle parse errors.

Many APIs offer a JSON mode that ensures the model's output is syntactically valid JSON. JSON mode guarantees valid syntax but does not necessarily guarantee that the required fields are present. A stronger option is schema-based structured output, where the developer supplies a JSON Schema that the output must follow.

Constrained decoding is the technique that makes schema guarantees possible. During generation, the system masks out any token that would violate the required format at that point. As a result, the model can only produce output that matches the grammar or schema.

A JSON Schema describes the expected fields, their types, which fields are required and any allowed values. For example, a schema might require a string field called name and an integer field called age. Enumerations can restrict a field to a fixed list of values, such as positive, negative or neutral.

Even with constrained decoding, the content inside the structure can still be wrong. The model might produce a perfectly formatted object containing an incorrect date or an invented value. Structure guarantees form, not truth, so content still needs evaluation.

Validation libraries are commonly used to check structured outputs after they are parsed. In Python, libraries such as Pydantic let developers define data models and validate responses against them. If validation fails, an application can retry the request, possibly including the error message so the model can fix its mistake.

## Embeddings

An embedding is a list of numbers, called a vector, that represents the meaning of a piece of text. Texts with similar meanings produce vectors that are close together in this numerical space. Texts with unrelated meanings produce vectors that are far apart.

Embeddings are created by embedding models, which are different from the chat models used to generate text. An embedding model takes text as input and returns a fixed-length vector as output. Common embedding sizes range from a few hundred to a few thousand dimensions.

The main advantage of embeddings is that they capture meaning rather than exact wording. A search for "how do I reset my password" can match a document titled "recovering account access" even though they share few words. This kind of search is called semantic search.

The similarity between two embeddings is usually measured with cosine similarity. Cosine similarity measures the angle between two vectors and ranges from minus one to one, where higher values mean more similar. Some systems use dot product or Euclidean distance instead, depending on how the embeddings were trained.

Embeddings from different models are not compatible with each other. If an application changes its embedding model, all stored embeddings must be regenerated with the new model. Mixing vectors from different models in the same index will produce meaningless similarity scores.

Generating embeddings is usually much cheaper and faster than generating text with a chat model. Applications often embed their whole document collection once in advance and store the vectors. At query time, only the user's question needs to be embedded.

The quality of embeddings varies by domain and language. A general-purpose embedding model may perform poorly on highly specialised text such as legal contracts or medical notes. Engineers should test candidate embedding models on examples from their own data before choosing one.

## Vector Databases and Similarity Search

A vector database is a storage system designed to save embeddings and quickly find the vectors most similar to a query vector. Each stored vector is usually accompanied by the original text and metadata such as source, date or author. When a query arrives, the database returns the closest matches along with their text and metadata.

Finding the most similar vectors is called nearest neighbour search. An exact search compares the query against every stored vector, which becomes slow when there are millions of vectors. To stay fast, most vector databases use approximate nearest neighbour algorithms, commonly abbreviated as ANN.

Approximate nearest neighbour algorithms trade a small amount of accuracy for a large gain in speed. A popular algorithm is HNSW, which stands for hierarchical navigable small world graphs. Other approaches include inverted file indexes and product quantisation, which compresses vectors to save memory.

The number of results returned by a similarity search is usually controlled by a parameter called k. Returning the top k results gives the application several candidate passages to work with. Choosing k involves a trade-off between recall and the amount of irrelevant text sent to the model.

Metadata filtering lets an application restrict a search to vectors that match certain conditions. For example, a search might be limited to documents from a particular department or created after a given date. Filtering is important for both relevance and access control.

Vector search is not always the best retrieval method on its own. Keyword search methods such as BM25 are better at matching exact terms like product codes, names and error messages. Hybrid search combines keyword and vector results to get the strengths of both.

A reranker is a model that reorders an initial list of search results by relevance. Rerankers are slower than vector search but more accurate, so they are typically applied only to the top few dozen candidates. Using a reranker often improves the quality of retrieved context noticeably.

## Chunking

Chunking is the process of splitting long documents into smaller pieces before embedding them. Each chunk is embedded and stored separately so that search can return only the relevant portion of a document. Chunking is necessary because embedding models have input limits and because whole documents are often too long to include in a prompt.

Chunk size is one of the most important settings in a retrieval system. Small chunks give precise matches but may lack the surrounding context needed to understand them. Large chunks preserve context but may dilute the embedding with unrelated content and waste tokens in the prompt.

Chunk overlap means repeating some text at the end of one chunk at the beginning of the next. Overlap reduces the chance that an important sentence is split across two chunks and lost. A typical overlap is a small fraction of the chunk size.

Fixed-size chunking splits text every set number of tokens or characters regardless of meaning. It is simple and predictable but can cut sentences or paragraphs in awkward places. Recursive chunking tries to split on natural boundaries such as paragraphs first, then sentences, then words.

Structure-aware chunking uses the document's own structure, such as headings, sections or code functions, to decide where to split. For a Markdown document, splitting at section headings often produces coherent chunks. For source code, splitting at function or class boundaries keeps related logic together.

Adding metadata to each chunk improves retrieval and citation. Useful metadata includes the source document title, section heading, page number and last updated date. Some systems prepend the document title or section heading to the chunk text itself so the embedding captures that context.

Poor chunking is one of the most common causes of weak retrieval-augmented generation results. If the right information is never retrieved, the model cannot use it no matter how good the prompt is. Inspecting retrieved chunks for real queries is one of the fastest ways to find chunking problems.

## Retrieval-Augmented Generation (RAG)

Retrieval-augmented generation, usually called RAG, is a pattern where an application retrieves relevant information and adds it to the prompt before the model generates an answer. RAG lets a model answer questions about private, specialised or recent information that it did not see during training. It is one of the most widely used patterns in AI engineering.

A basic RAG pipeline has two phases: indexing and querying. During indexing, documents are loaded, split into chunks, embedded and stored in a vector index. During querying, the user's question is embedded, similar chunks are retrieved, and those chunks are inserted into the prompt along with the question.

The prompt in a RAG system usually instructs the model to answer using only the provided context. It often also tells the model to say it does not know when the context does not contain the answer. These instructions reduce the chance that the model fills gaps with invented information.

RAG has several advantages over relying on the model's built-in knowledge. Knowledge can be updated by changing the documents rather than retraining the model. Answers can include citations to the source documents, which helps users verify them.

RAG quality depends on both retrieval and generation. If retrieval returns irrelevant chunks, the model may answer incorrectly or refuse to answer. If retrieval succeeds but the prompt is poor, the model may ignore or misread the context.

Engineers therefore evaluate retrieval and generation separately. Retrieval metrics include context precision, which checks whether retrieved chunks are relevant, and context recall, which checks whether all needed information was retrieved. Generation metrics include faithfulness, which checks whether the answer is supported by the context, and answer relevancy, which checks whether it addresses the question.

Conversational RAG must handle follow-up questions that depend on earlier messages. A question like "what about the second one?" cannot be searched meaningfully on its own. A common solution is to have the model rewrite the follow-up into a standalone question before retrieval.

## Hallucinations and Grounding

A hallucination is an output from a language model that sounds plausible but is false, unsupported or invented. Examples include made-up citations, incorrect dates, fictional product features and quotes that were never said. Hallucinations are one of the biggest obstacles to using language models in high-stakes settings.

Hallucinations happen because language models generate statistically likely text rather than retrieving verified facts. When the model lacks the right knowledge, it may still produce a confident answer that fits the pattern of a correct one. Models are not inherently aware of the boundary between what they know and what they are guessing.

Grounding means tying the model's answers to a trusted source of information. Retrieval-augmented generation is the most common grounding technique, because it provides the model with relevant source text. Tool calls that fetch live data from databases or APIs are another form of grounding.

Instructions in the prompt can reduce hallucinations. Telling the model to answer only from the provided context and to admit uncertainty helps it avoid guessing. Asking the model to quote or cite the supporting passage makes unsupported claims easier to spot.

Lower sampling temperatures can reduce some random errors but do not eliminate hallucinations. A model can be confidently wrong even at temperature zero. Temperature controls randomness, not factual accuracy.

Faithfulness is a metric that measures whether an answer's claims are supported by the provided context. A faithful answer may still be wrong if the context itself is wrong. Correctness, by contrast, measures whether the answer matches the true answer.

No current technique removes hallucinations completely. AI engineers should design systems assuming that some errors will occur and ensure that mistakes are caught or have limited impact. Human review remains important for high-risk outputs such as medical, legal and financial advice.

## Tool Calling and Function Calling

Tool calling, also known as function calling, lets a language model request that the application run a specific function with specific arguments. The model itself does not execute code; it produces a structured request, and the application decides whether and how to run it. This lets a model fetch live information or take actions in other systems.

To use tool calling, the developer describes each available tool to the model. A tool description usually includes a name, a natural-language explanation of what it does, and a JSON Schema describing its parameters. The model uses these descriptions to decide which tool fits the user's request.

A typical tool-calling exchange happens in several steps. First, the user asks a question, such as what the weather is in a particular city. The model responds with a tool call naming a weather function and a city argument, and the application runs that function and sends the result back to the model. Finally, the model uses the result to write a natural-language answer for the user.

Tool results are usually sent back in a message with a tool role, linked to the original call by an identifier. The model can then decide to answer, ask a clarifying question or call another tool. Some models can request several tool calls at once, which is called parallel tool calling.

Good tool descriptions are as important as good prompts. A clear name and description help the model choose the right tool and fill in arguments correctly. Ambiguous or overlapping tools often lead to wrong tool choices.

Applications should always validate the arguments a model provides before running a tool. The model can produce invalid values, missing fields or arguments that a user maliciously influenced. Treating model-generated arguments as untrusted input is a basic safety practice.

Tools that only read information carry less risk than tools that change things. Actions such as sending emails, making payments or deleting records should often require explicit human confirmation. This pattern is commonly called human in the loop.

## Agents and the ReAct Loop

An AI agent is a system in which a language model decides, step by step, what actions to take in order to accomplish a goal. Unlike a single prompt and response, an agent can take multiple steps, observe the results, and adjust its plan. Agents are typically built from a model, a set of tools, instructions and a control loop.

The ReAct pattern is a widely used approach for building agents. ReAct stands for reasoning and acting, and it alternates between thinking about what to do and taking an action. After each action, the agent observes the result and uses it to decide the next step.

A ReAct loop usually follows a simple cycle of thought, action and observation. In the thought step, the model reasons about the current state and what information it still needs. In the action step, it calls a tool, and in the observation step, it reads the tool's output.

The loop continues until the model decides that it has enough information to give a final answer. Because the model controls when to stop, the application should also enforce a maximum number of steps. Without a step limit, an agent can loop indefinitely and waste time and money.

Agents are useful for open-ended tasks where the required steps are not known in advance. Examples include researching a question across several sources, debugging code, or completing a multi-step booking. For tasks with fixed, predictable steps, a simple hard-coded workflow is usually more reliable and cheaper.

A workflow is a system where the developer defines the sequence of model calls and tool calls in code. An agent is a system where the model itself decides the sequence dynamically. Many practical applications combine both, using fixed workflows for most steps and agentic flexibility only where needed.

Evaluating agents involves checking both the final outcome and the path taken. Outcome checks ask whether the task was completed correctly. Trajectory checks ask whether the agent used appropriate tools, avoided unnecessary steps and stayed within safety limits.

## Memory and Conversation History Management

Language models are stateless, which means they do not remember anything between separate API requests. To create the experience of an ongoing conversation, the application must send the relevant conversation history with every new request. This history usually consists of earlier user and assistant messages.

Short-term memory refers to the conversation history kept within the current session. The simplest approach is to send the entire history every time. This works for short conversations but eventually exceeds the context window and becomes expensive.

A sliding window keeps only the most recent messages and drops older ones. This approach is simple and keeps the token count bounded. Its weakness is that important details from early in the conversation may be forgotten.

Summarisation memory periodically compresses older messages into a short summary. The summary is included in place of the full older messages, while recent messages are kept in full. This preserves key facts while saving tokens, although some detail is inevitably lost.

Long-term memory stores information across sessions so the assistant can recall it later. Examples include a user's preferences, past decisions or important facts they have shared. Long-term memories are often stored in a database and retrieved when relevant, sometimes using embeddings.

Users should be able to see and control what an assistant remembers about them. Clear controls for viewing, editing and deleting memories support privacy and trust. Many privacy regulations also require that personal data can be deleted on request.

Memory management directly affects cost and latency. Every token of history sent with a request is paid for and processed again. Efficient memory strategies keep conversations coherent without sending unnecessary text.

## Fine-tuning vs Prompting vs RAG

Prompting, retrieval-augmented generation and fine-tuning are the three main ways to adapt a language model to a task. Each approach solves a different kind of problem, and they are often combined. Choosing among them is one of the most common decisions in AI engineering.

Prompting changes the instructions and examples sent to the model without changing the model itself. It is fast, cheap and easy to iterate on, which makes it the recommended starting point. Many tasks can be solved well with prompting alone.

Retrieval-augmented generation supplies the model with external knowledge at request time. It is the best choice when the model needs facts it does not know, such as private company documents or frequently changing information. RAG also makes it easy to update knowledge and to cite sources.

Fine-tuning continues training a pretrained model on a smaller, task-specific dataset. It changes the model's weights so that the desired behaviour becomes built in. Fine-tuning is best for teaching consistent style, format, tone or narrow skills that are hard to describe in a prompt.

A common misconception is that fine-tuning is the best way to teach a model new facts. In practice, fine-tuning is usually less reliable than RAG for adding knowledge and does not provide citations. Fine-tuned knowledge also becomes stale and requires retraining to update.

Parameter-efficient fine-tuning methods train only a small number of extra parameters instead of the whole model. A popular method is LoRA, which stands for low-rank adaptation. These methods reduce the memory and compute required for fine-tuning.

A practical rule is to start with prompting, add RAG when knowledge is missing, and consider fine-tuning when behaviour or format remains inconsistent. Evaluation results should drive each step rather than assumptions. Combining a fine-tuned model with RAG is common in mature systems.

## Evaluation: Golden Datasets, LLM-as-a-Judge and Metrics

Evaluation is the practice of systematically measuring how well an AI system performs. Because model outputs vary and are hard to predict, evaluation is the main way to know whether a change made things better or worse. Many practitioners consider evaluation the most important skill in AI engineering.

A golden dataset is a curated set of test inputs paired with expected outputs or reference answers. Each item in the dataset is often called a golden or a test case. Golden datasets act like unit tests for AI systems and allow changes to be compared fairly.

Good golden datasets cover typical questions, edge cases and known failure modes. They should include questions that the system should refuse or answer with uncertainty, not only questions with clear answers. Even a few dozen well-chosen examples are far more useful than none.

Golden datasets can be written by hand, collected from real user queries, or generated synthetically with a language model. Synthetic generation is fast and scalable but can produce unrealistic or incorrect examples. Generated test cases should be reviewed by a human before they are trusted.

LLM-as-a-judge is a technique where a language model grades the output of another model. The judge receives the question, the answer and often a reference answer or rubric, and returns a score or a pass or fail verdict. This approach scales evaluation of open-ended text far beyond what humans can review.

LLM judges have known biases that engineers should account for. They may prefer longer answers, favour the first option in a comparison, or rate outputs from similar models more highly. Clear rubrics, binary pass or fail criteria and requests for reasoning before the verdict help reduce these problems.

Common metrics for question-answering systems include correctness, faithfulness, answer relevancy and completeness. For retrieval, common metrics include context precision, context recall and hit rate. For agents, metrics often include task success rate, number of steps and tool-call accuracy.

## Observability and Tracing

Observability is the ability to understand what a system is doing internally by examining the data it produces. For AI applications, observability means recording prompts, responses, tool calls, latency, token usage and errors. Without it, engineers cannot easily explain why a model gave a particular answer.

A trace is a record of everything that happened while handling a single request. A trace is made up of spans, where each span represents one step such as a retrieval, a model call or a tool execution. Spans are nested to show which steps happened inside which larger operations.

For a RAG application, a trace might show the user question, the rewritten query, the retrieved chunks, the final prompt and the model's answer. Seeing all of these together makes it easy to tell whether a bad answer came from poor retrieval or poor generation. This kind of root-cause analysis is the main benefit of tracing.

Each model call in a trace usually records the model name, parameters, input tokens, output tokens, latency and cost. Aggregating these values across many traces shows overall usage and spending. Sudden changes in these numbers often signal a bug or a change in user behaviour.

Observability tools often let teams attach scores to traces. Scores can come from user feedback such as thumbs up or thumbs down, from automated evaluators, or from human reviewers. Linking quality scores to traces connects evaluation directly to real usage.

Production traces are a valuable source of test cases. Interesting or failing traces can be added to a golden dataset so that the problem is checked in future evaluations. This creates a feedback loop between production monitoring and offline testing.

Because traces contain prompts and responses, they may include personal or sensitive data. Teams should apply masking, access controls and retention limits to trace data. Observability must be balanced with the privacy obligations covered in the privacy section.

## Latency, Cost and Token Budgets

Latency is the time a user waits for a response, and it strongly affects how an AI application feels to use. Two latency measures are especially important: time to first token and total response time. Time to first token is how long it takes before any output appears, while total time includes generating the entire response.

Output tokens are usually generated much more slowly than input tokens are processed. As a result, long answers take noticeably longer to produce than short ones. Asking for concise outputs is one of the simplest ways to reduce latency.

Model size also affects latency, because larger models generally generate tokens more slowly than smaller ones. Many systems route simple requests to a small, fast model and reserve a larger model for difficult requests. This strategy is often called model routing or model cascading.

The cost of a hosted model request is typically based on the number of input and output tokens. Output tokens are usually priced higher than input tokens. Cost per request therefore depends on prompt length, retrieved context, conversation history and answer length.

A token budget is a planned limit on how many tokens each part of a request may use. Budgets prevent costs from growing unexpectedly as conversations get longer or more documents are retrieved. They also help ensure requests stay within the context window.

Caching can reduce both latency and cost. Response caching stores answers to repeated identical requests so the model does not need to run again. Prompt caching, offered by some providers, reuses the processing of a repeated prompt prefix such as a long system prompt.

Agents and multi-step workflows multiply costs because each step makes another model call. An agent that takes ten steps may cost ten times as much as a single call, plus the growing history at each step. Step limits and efficient tool design help keep agent costs under control.

## Safety: Prompt Injection and Guardrails

Prompt injection is an attack in which someone crafts input that causes a language model to ignore its intended instructions. A direct prompt injection comes from the user, for example by typing "ignore all previous instructions." An indirect prompt injection hides malicious instructions inside content the model reads, such as a web page, email or document.

Indirect prompt injection is particularly dangerous for applications that use retrieval or tools. A model that reads an attacker's web page might follow hidden instructions to reveal private data or call a harmful tool. The risk grows when a model has access to sensitive data, untrusted content and the ability to send data outward at the same time.

There is currently no complete defence against prompt injection. Language models process instructions and data in the same stream of tokens, so they cannot perfectly separate the two. Defences therefore focus on reducing risk and limiting the damage when an injection succeeds.

The principle of least privilege is a core defence. An AI application should have only the permissions and tools it truly needs for its task. Limiting tool access means that even a successful injection can do little harm.

Human approval for high-impact actions adds another layer of protection. If the model must ask a person before sending money, deleting data or emailing external addresses, a hijacked model cannot complete those actions alone. This approach trades some convenience for much greater safety.

Guardrails are checks placed around a model to prevent unsafe inputs or outputs. Input guardrails screen user messages for attacks, abuse or off-topic requests before they reach the model. Output guardrails check the model's response for harmful content, leaked secrets or policy violations before it reaches the user.

Safety should be tested deliberately rather than assumed. Red teaming means intentionally trying to break a system using adversarial prompts and scenarios. Findings from red teaming should be added to the evaluation dataset so that fixes are checked over time.

## Privacy and Data Handling

AI applications frequently process personal and sensitive information, so privacy must be considered from the start. Personal data includes names, email addresses, phone numbers, locations and any information that can identify a person. Personally identifiable information is commonly abbreviated as PII.

Every piece of data sent to a hosted model leaves the application's own infrastructure. Engineers should understand how the provider stores, logs and uses that data. Many providers offer business terms stating that customer data will not be used to train their models, but these terms should be checked rather than assumed.

Data minimisation means sending only the data that is necessary for the task. If a model needs only a customer's question, there is no reason to include their full account record. Minimising data reduces both privacy risk and token cost.

Redaction and masking remove or replace sensitive values before data is sent to a model or stored in logs. For example, an email address might be replaced with a placeholder token. Some systems restore the original values after the model responds if they are needed in the final output.

Logs, traces and evaluation datasets can quietly accumulate sensitive data. Retention policies should define how long such data is kept and when it is deleted. Access to these records should be restricted to people who need it.

Access control is essential in retrieval systems. A RAG application must make sure users can retrieve only documents they are allowed to see. Applying permission filters at retrieval time prevents the model from leaking restricted content in its answers.

Fine-tuning on private data creates special risks. A model trained on sensitive records may reproduce fragments of that data in its outputs. Data used for training should be cleaned of sensitive information unless there is a strong reason and proper safeguards.

## Deployment and APIs: Rate Limits, Retries and Streaming

Most AI applications access language models through an application programming interface, or API, provided by a model vendor or a self-hosted server. The application sends an HTTP request containing messages and parameters and receives the model's response. API keys are used to authenticate these requests.

API keys must be kept secret and should never be placed in client-side code, public repositories or prompts. They are normally stored in environment variables or a secrets manager on the server. If a key is exposed, it should be revoked and replaced immediately.

Rate limits restrict how many requests or tokens an account can use within a time period. Limits are often expressed as requests per minute and tokens per minute. When an application exceeds a rate limit, the API returns an error, commonly with the HTTP status code 429.

Retries are the standard response to temporary errors such as rate limits, timeouts and server errors. Retries should use exponential backoff, which means waiting progressively longer between attempts. Adding random jitter to the wait time prevents many clients from retrying at exactly the same moment.

Streaming sends the model's output to the client token by token as it is generated, instead of waiting for the whole response. Streaming does not make total generation faster, but it greatly improves perceived speed because users see text appear immediately. Most chat interfaces use streaming for this reason.

Fallbacks improve reliability by switching to a different model or provider when the primary one fails. Fallback models should be evaluated in advance so their quality is known. Some teams use a gateway or proxy layer to manage routing, fallbacks and usage tracking in one place.

Model versions change over time, and providers eventually retire older models. Pinning a specific model version in production prevents unexpected behaviour changes. Before upgrading, teams should rerun their evaluation datasets to confirm that quality has not regressed.

## Glossary Recap

This section restates key terms in short definitions for quick reference. An agent is a system where a language model chooses a sequence of actions and tool calls to reach a goal. Attention is the transformer mechanism that lets a model weigh the relevance of every token to every other token.

Chunking is splitting documents into smaller pieces before embedding and indexing them. A context window is the maximum number of tokens a model can process in a single request, including both input and output. Constrained decoding restricts generation so that the output must match a required format or schema.

An embedding is a vector of numbers that represents the meaning of text or other content. Fine-tuning is further training of a pretrained model on task-specific examples to change its behaviour. A golden dataset is a curated set of test inputs with expected outputs used for evaluation.

Grounding is connecting a model's answers to trusted sources of information. A hallucination is a confident model output that is false or unsupported. LLM-as-a-judge is the use of a language model to score or grade other model outputs.

Prompt injection is an attack that smuggles instructions into a model's input to override its intended behaviour. RAG, or retrieval-augmented generation, retrieves relevant information and adds it to the prompt before generation. ReAct is an agent pattern that alternates between reasoning, taking an action and observing the result.

A system prompt is a developer-written message that sets an assistant's behaviour and rules. Temperature is a sampling parameter where lower values give more predictable output and higher values give more varied output. Top-p sampling limits token choices to the most probable tokens whose combined probability reaches a threshold.

A token is the basic unit of text that a model reads and writes, often a fragment of a word. Tool calling is a feature that lets a model request the execution of a defined function with structured arguments. A trace is a record of every step taken while handling a single request, made up of nested spans.

A vector database stores embeddings and supports fast similarity search over them. Cosine similarity is a common measure of how close two embeddings are in meaning. Faithfulness is an evaluation metric that checks whether an answer is supported by its provided context.
