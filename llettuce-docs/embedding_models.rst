Embedding Models
=================

This page contains details about the embedding models used in the project.

.. _embedding-models:

Embedding Models Documentation
------------------------------

This section contains details about the embedding models used in the project.


Available Models:
~~~~~~~~~~~~~~~~~

1. (Bidirectional Gated Encoder - Small)

BGESMALL
--------
    - Model Name: Bidirectional Gated Encoder
    
    - Version: Small
    
    - Dimensions: 384
    
    - Description:
        BGE-small has been designed for tasks such as sentence
        embedding, semantic similarity, and information retrieval. 
        
    - Benefits:
        BGE-small more efficient in terms of speed and memory,
        while still being capable of generating high-quality
        sentence or document embeddings.
        
    - Research Paper:
        arXiv:2402.03216 



2. Sentence-BERT MiniLM

MINILM
------
    - Model Name: Sentence-BERT
    
    - Version: MiniLM
    
    - Dimensions: 384
    
    - Description:
        MiniLM is a smaller, more efficient version of the 
        Sentence-BERT (SBERT) model, specifically optimized 
        for tasks such as semantic similarity, sentence 
        embeddings, and information retrieval.
        
    - Benefits:
        The MiniLM variant provides a good trade-off between 
        computational efficiency and performance, making it
        suitable for use cases where speed and resource 
        limitations are important, without sacrificing 
        too much accuracy.
        
    - Research Paper:
        arXiv:1908.10084



3. Generalizable T5 Retrieval - Base

GTR_T5_BASE
-----------
    - Model Name: Generalizable T5 Retrieval
    
    - Version: Base
    
    - Dimensions: 768
    
    - Description:
        GTR-T5 is a dense retrieval model using a dual encoder 
        architecture for efficient semantic search and passage 
        retrieval. It encodes queries and documents separately
        into a shared embedding space, allowing fast and scalable
        retrieval using a dot-product similarity. This model 
        is based on T5 and optimized for generalization 
        across diverse tasks.

    - Benefits:
        GTR-T5 is highly generalizable across different tasks, scalable
        in size, and efficient for large-scale retrieval with precomputed 
        document embeddings. 
        
        It leverages the T5 model for deep semantic understanding and 
        ensures fast retrieval using Approximate Nearest Neighbor 
        (ANN) search, making it both powerful and efficient 
        for various retrieval tasks.
        
    - Research Paper:
        ArXiv:abs/2112.07899



4. Generalizable T5 Retrieval (Large)

GTR_T5_LARGE
------------
    - Model Name: Generalizable T5 Retrieval
    
    - Version: Large
    
    - Dimensions: 1024
    
    - Description:
        GTR-T5-Large is a powerful version of the Generalizable T5 
        Retrieval model designed for dense retrieval tasks. 
        
        It encodes queries and documents into a shared embedding 
        space to enable efficient retrieval. The large version 
        enhances performance by offering more capacity for 
        complex semantic understanding.
        
    - Benefits:
        GTR-T5-Large excels at generalizing across various tasks and domains.
        It is scalable and ideal for handling large datasets while 
        maintaining efficient retrieval. Leveraging T5’s deep 
        language understanding, it supports fast retrieval with 
        Approximate Nearest Neighbor (ANN) search, making it 
        highly effective for large-scale semantic search 
        and retrieval tasks.
        
    - Research Paper:
        ArXiv:abs/2112.07899



5. Embedding Models for Search Engines (Base)

E5_BASE
-------
    - Model Name: Embedding Models for Search Engines
    
    - Version: Base
    
    - Dimensions: 768
    
    - Description:
        E5 is a family of dense retrieval models by Microsoft, designed
        to generate high-quality text embeddings for search and 
        retrieval tasks. 
        
        It leverages contrastive learning on multilingual text pairs,
        combined with supervised fine-tuning, to perform well in 
        zero-shot and fine-tuned settings. 
        
        The base version provides efficient embeddings for tasks
        like semantic search, passage retrieval, document 
        ranking, and clustering.

    - Benefits:
        E5 offers high-quality semantic embeddings that generalize well across
        different domains and tasks. 
        
        Fine-tuned on the BEIR benchmark, it excels in cross-domain 
        retrieval and semantic search scenarios. 
        
        E5 also supports instruction-tuned variants for enhanced 
        task-specific performance, and demonstrates strong
        results on retrieval benchmarks like BEIR and MTEB.
        
    - Research Paper:
        arXiv:2212.03533



6. Embedding Models for Search Engines (Large)

E5_LARGE
--------
    - Model Name: Embedding Models for Search Engines
    
    - Version: Large
    
    - Dimensions: 1024
    
    - Description:
        E5-Large is an advanced version of Microsoft’s E5 family of 
        dense retrieval models, designed for generating high-quality 
        text embeddings for search, retrieval, and ranking tasks. 
        
        Like E5-Base, it utilizes contrastive learning with multilingual
        text pairs and fine-tuning on supervised datasets, 
        but the large version offers greater capacity, 
        improving performance on more complex tasks.
        
    - Benefits:
        E5-Large provides deeper semantic understanding due to its 
        larger model size, offering improved performance on 
        retrieval tasks across diverse domains. 
        
        It excels in semantic search, cross-domain retrieval,
        and document ranking, leveraging its larger capacity
        for better generalization and accuracy. 
        
        E5-Large demonstrates strong results on benchmarks 
        such as BEIR and MTEB.
        
    - Research Paper:
        arXiv:2212.03533



7. DistilBERT (Uncased)

DISTILBERT_BASE_UNCASED
-----------------------
    - Model Name: DistilBERT
    
    - Version: Base Uncased
    
    - Dimensions: 768
    
    - Description:
        DistilBERT is a smaller, faster, and lighter version of 
        the BERT model designed by Hugging Face for NLP tasks. 
        
        It offers 97% of BERT's performance but is 40% smaller, 
        making it ideal for deployment in resource-constrained
        environments. 
        
        DistilBERT reduces computational overhead, enabling faster
        inference while retaining high accuracy on most tasks.
        
    - Benefits:
        DistilBERT is 40% smaller and 60% faster than BERT, requiring 
        fewer resources while retaining 97% of BERT’s performance, 
        making it ideal for efficient deployment and easy 
        fine-tuning in resource-constrained environments.
        
    - Research Paper:
        arXiv:1910.01108
    


8. distiluse-base-multilingual

DISTILUSE_BASE_MULTILINGUAL
---------------------------
    - Model Name: distiluse-base-multilingual-cased-v1
    
    - Version: Base Multilingual
    
    - Dimensions: 512
    
    - Description:
        DistilUSE is a multilingual variant of the DistilBERT model 
        by Hugging Face, optimized for generating high-quality 
        multilingual text embeddings. 
        
        It is pre-trained on a large-scale multilingual corpus, 
        enabling it to encode text from multiple languages 
        into a shared embedding space.
        
    - Benefits:
        DistilUSE offers high-quality multilingual embeddings that 
        generalize well across different languages and tasks. 
        
        It is efficient for cross-lingual search, semantic similarity, 
        and retrieval tasks, making it ideal for multilingual 
        applications and scenarios.
        
    - Research Paper:
        ArXiv. /abs/1910.01108



9. Contriever 

CONTRIEVER
----------
    - Model Name: Contriever
    
    - Version: Contriever

    - Dimensions: 768
    
    - Description:
        Contriever, developed by Facebook, is an unsupervised dense 
        retrieval model designed for semantic search and information
        retrieval tasks without the need for labeled data.
        
        Using contrastive learning, it generates high-quality text
        embeddings for tasks like zero-shot retrieval, making it
        effective in domains where no task-specific 
        data is available. 
        
    - Benefits:
        Contriever excels at unsupervised dense retrieval, offering
        strong zero-shot performance across various domains 
        using contrastive learning, and is highly versatile,
        achieving good results in fields like biomedical,
        legal, and scientific datasets without 
        task-specific supervision.
        
    - Research Paper:
        ArXiv. /abs/2112.09118


