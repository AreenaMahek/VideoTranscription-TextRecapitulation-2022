import logging
from gensim.summarization.pagerank_weighted import pagerank_weighted as _pagerank
from gensim.summarization.textcleaner import clean_text_by_sentences as _clean_text_by_sentences
from gensim.summarization.commons import build_graph as _build_graph
from gensim.summarization.commons import remove_unreachable_nodes as _remove_unreachable_nodes
from gensim.summarization.bm25 import get_bm25_weights as _bm25_weights
from gensim.corpora import Dictionary
from math import log10 as _log10
from six.moves import xrange


INPUT_MIN_LENGTH = 5

WEIGHT_THRESHOLD = 1.e-3  #1 × 10−5. In other words, 0.00001

logger = logging.getLogger(__name__)


def _set_graph_edge_weights(graph):
    documents = graph.nodes()
    weights = _bm25_weights(documents)

    for i in xrange(len(documents)):
        for j in xrange(len(documents)):
            if i == j or weights[i][j] < WEIGHT_THRESHOLD:
                continue

            sentence_1 = documents[i]
            sentence_2 = documents[j]

            edge_1 = (sentence_1, sentence_2)
            edge_2 = (sentence_2, sentence_1)

            if not graph.has_edge(edge_1):
                graph.add_edge(edge_1, weights[i][j])
            if not graph.has_edge(edge_2):
                graph.add_edge(edge_2, weights[j][i])

    # Handles the case in which all similarities are zero.
    # The resultant summary will consist of random sentences.
    if all(graph.edge_weight(edge) == 0 for edge in graph.edges()):
        _create_valid_graph(graph)


def _create_valid_graph(graph):
    nodes = graph.nodes()

    for i in xrange(len(nodes)):
        for j in xrange(len(nodes)):
            if i == j:
                continue

            edge = (nodes[i], nodes[j])

            if graph.has_edge(edge):
                graph.del_edge(edge)

            graph.add_edge(edge, 1)


def _get_doc_length(doc):
    return sum([item[1] for item in doc])


def _get_similarity(doc1, doc2, vec1, vec2):
    numerator = vec1.dot(vec2.transpose()).toarray()[0][0]
    length_1 = _get_doc_length(doc1)
    length_2 = _get_doc_length(doc2)

    denominator = _log10(length_1) + _log10(length_2) if length_1 > 0 and length_2 > 0 else 0

    return numerator / denominator if denominator != 0 else 0


def _build_corpus(sentences):
    split_tokens = [sentence.token.split() for sentence in sentences]
    dictionary = Dictionary(split_tokens)
    return [dictionary.doc2bow(token) for token in split_tokens]


def _get_important_sentences(sentences, corpus, important_docs):
    hashable_corpus = _build_hasheable_corpus(corpus)
    sentences_by_corpus = dict(zip(hashable_corpus, sentences))
    return [sentences_by_corpus[tuple(important_doc)] for important_doc in important_docs]


def _get_sentences_with_word_count(sentences, word_count):
    """ Given a list of sentences, returns a list of sentences with a
    total word count similar to the word count provided."""
    length = 0
    selected_sentences = []

    # Loops until the word count is reached.
    for sentence in sentences:
        words_in_sentence = len(sentence.text.split())

        # Checks if the inclusion of the sentence gives a better approximation
        # to the word parameter.
        if abs(word_count - length - words_in_sentence) > abs(word_count - length):
            return selected_sentences

        selected_sentences.append(sentence)
        length += words_in_sentence

    return selected_sentences


def _extract_important_sentences(sentences, corpus, important_docs, word_count):
    important_sentences = _get_important_sentences(sentences, corpus, important_docs)

    # If no "word_count" option is provided, the number of sentences is
    # reduced by the provided ratio. Else, the ratio is ignored.
    return important_sentences if word_count is None else _get_sentences_with_word_count(important_sentences, word_count)


def _format_results(extracted_sentences, split):
    if split:
        return [sentence.text for sentence in extracted_sentences]
    return "\n".join([sentence.text for sentence in extracted_sentences])


def _build_hasheable_corpus(corpus):
    return [tuple(doc) for doc in corpus]


def summarize_corpus(corpus, ratio=0.2):
    """
    Returns a list of the most important documents of a corpus using a
    variation of the TextRank algorithm.
    The input must have at least INPUT_MIN_LENGTH (%d) documents for the
    summary to make sense.

    The length of the output can be specified using the ratio parameter,
    which determines how many documents will be chosen for the summary
    (defaults at 20%% of the number of documents of the corpus).

    The most important documents are returned as a list sorted by the
    document score, highest first.

    """ % INPUT_MIN_LENGTH
    hashable_corpus = _build_hasheable_corpus(corpus)

    # If the corpus is empty, the function ends.
    if len(corpus) == 0:
        logger.warning("Input corpus is empty.")
        return

    # Warns the user if there are too few documents.
    if len(corpus) < INPUT_MIN_LENGTH:
        logger.warning("Input corpus is expected to have at least " + str(INPUT_MIN_LENGTH) + " documents.")

    graph = _build_graph(hashable_corpus)
    _set_graph_edge_weights(graph)
    _remove_unreachable_nodes(graph)

    pagerank_scores = _pagerank(graph)

    hashable_corpus.sort(key=lambda doc: pagerank_scores.get(doc, 0), reverse=True)

    return [list(doc) for doc in hashable_corpus[:int(len(corpus) * ratio)]]



def summarize(text, ratio=0.3, word_count=None, split=False):
    """
    Returns a summarized version of the given text using a variation of
    the TextRank algorithm.
    The input must be longer than INPUT_MIN_LENGTH sentences for the
    summary to make sense and must be given as a string.

    The output summary will consist of the most representative sentences
    and will also be returned as a string, divided by newlines. If the
    split parameter is set to True, a list of sentences will be
    returned.

    The length of the output can be specified using the ratio and
    word_count parameters:
        ratio should be a number between 0 and 1 that determines the
    percentage of the number of sentences of the original text to be
    chosen for the summary (defaults at 0.2).
        word_count determines how many words will the output contain.
    If both parameters are provided, the ratio will be ignored.
    """
    # Gets a list of processed sentences.
    sentences = _clean_text_by_sentences(text)

    # If no sentence could be identified, the function ends.
    if len(sentences) == 0:
        logger.warning("Input text is empty.")
        return

    # Warns if the text is too short.
    if len(sentences) < INPUT_MIN_LENGTH:
        logger.warning("Input text is expected to have at least " + str(INPUT_MIN_LENGTH) + " sentences.")

    corpus = _build_corpus(sentences)

    most_important_docs = summarize_corpus(corpus, ratio=ratio if word_count is None else 1)

    # Extracts the most important sentences with the selected criterion.
    extracted_sentences = _extract_important_sentences(sentences, corpus, most_important_docs, word_count)

    # Sorts the extracted sentences by apparition order in the original text.
    extracted_sentences.sort(key=lambda s: s.index)
    final_text= _format_results(extracted_sentences, split)
    return final_text


# text="The broad range of Pacific Alaskan salmon has resulted in the creation of a complex and multiorganizational system of management that includes the state of Alaska, various federal departments, a Congressionally-mandated fishery council, and a number of commercial and nongovernmental fish organizations. In the Bering Sea salmon are caught by the commercial groundfish fleet as by-catch. On the Yukon River salmon are commercially and traditionally harvested for both economic and cultural sustenance by the Yup’ik residents of the Yukon Delta. Declining salmon populations has driven scientific research which considers the effects of Bering Sea salmon by-catch. My research findings indicate that Bering Sea fisheries occur where juvenile salmon mature, directly impacting Yukon River salmon populations. Further, the research reflects that although Yukon salmon populations have plummeted, a recent effort was made to open the northern Bering Sea, which includes the Yukon River coastal shelf, to deep-sea commercial fishing. By researching the relationship of policy to cultural salmon dependence, it becomes evident that Alaskan salmon-tribes are excluded from salmon management and decision-making. Legal research reflects that three basic federal Indian concepts – inherent rights, Indian Country, and tribal right of occupancy – emerge as potential foundations that may allow Alaskan salmontribes to begin sharing legal responsibility over salmon. Yukon River salmon are an international and anadromous species that require multiorganizational management. My research reflects that current management favors the Bering Sea commercial fishing industry, despite data indicating Bering Sea fisheries impact Yukon salmon populations and an overall downward trend in Yukon salmon populations."
# "Using GIS Site Suitability Analysis to Study Adaptability and Evolution of Life: Locating Springs in Mantle Units of Ophiolites Alexandrea Bowman, University of Rhode Island GIS is a powerful tool that can be used to locate springs sourced in ophiolites. The unique features associated with these springs include a reducing subsurface environment reacting at low temperatures producing high pH, Ca-rich formation fluids with high dissolved hydrogen and methane. Because of their unique chemical characteristics, these areas are often associated with microbes and are thought to be similar to the features that enabled life to evolve on Earth. Locating and sampling these springs could offer a deeper look into Earth's deep biosphere and the history of life on Earth. Springs have tradiitionally been located using expensive and time consuming field techniques. Field work can be dangerous. The goal of this study was to develop a model that could locate these unique geological features without first going into the field, thus saving time, money and reducing the risks associated with remote field localities. A GIS site suitability analysis works by overlaying existing geo-referenced data into a computer program and adding the different data sets after assigning a numerical value to the important fields. For this project, I used surface and ground water maps, geologic maps, a soil map, and a fault map for four counties in Northern California. The model has demonstrated that it is possible to use this time of model and apply it to a complex geologic area to produce a usable field map for future field work."
# "Ferritin is a ubiquitous iron storage and detoxification protein found highly conserved in species from bacteria to plants to humans. In mammals, ferritin is composed of two functionallyand genetically distinct subunit types, H (heavy, ~21,000 Da) and L (light, ~19,000 Da) subunits which co-assemble in various ratios with tissue specific distribution to form a shell-like protein. The H-subunit is responsible for the fast conversion of Fe(II) to Fe(III) by dioxygen (or H2O2) whereas the L-subunit is thought to contribute to the nucleation of the iron core. In the present work, we investigated the iron oxidation and deposition mechanism in two recombinant heteropolymers ferritin samples of ~20H:4L (termed H/L) and ~22L:2H (termed L/H) ratios. Data indicates that iron oxidation occurs mainly on the H-subunit with a stoichiometry of 2Fe(II):1O2, suggesting formation of H2O2. The H/L sample completely regenerates its ferroxidase activity within a short period of time suggesting rapid movement of Fe(III) from the ferroxidase center to the cavity to form the mineral core, consistent with the role of L-chain in facilitating iron turn-over at the ferroxidase center of the H-subunit. In L/H, Fe(II) oxidation and mineralization appears to occur by two simultaneous pathways at all levels of iron additions: a ferroxidation pathway with a 2Fe(II)/1O2 ratio and a mineralization pathway with a 4Fe(II)/1O2 resulting in an average net stoichiometry of ~3Fe(II)/1O2. These results illustrate how recombinant heteropolymer ferritins control iron and oxygen toxicity while providing a safe reservoir for reversible uptake and release of iron for use by the cell."
# "An Assessment of Oral Health on the Pine Ridge Indian Reservation Joaquin R Gallegos, Terry Batliner, DDS, MBA, John T Brinton, MS, Dallas M Daniels, RDH, BS, Anne Wilson, DDS, MS, Maxine Janis, MPH, RDH, Kimberly E Lind, MPH, Deborah H Glueck, PhD, Judith Albino, PhD. Centers for American Indian and Alaska Native Health, University of Colorado, Colorado School of Public Health We assessed the oral health of the Pine Ridge Oglala Lakota people, described a new oral health assessment tool for Indigenous people, and suggested ways to improve Native oral health. The Check Up Study team of dentist and dental hygienists performed examinations of teeth and oral soft tissue for a convenience sample of 292 adults and children. Screening personnel counted the number of decayed, filled, sealed and total teeth, used probes to measure periodontal disease, and screened for oral lesions. Half of adults had 27 or fewer teeth. Sixteen percent of adults had at least one tooth with a pocket depth > 6mm. Participants had higher numbers of decayed teeth (p"
# "The purpose of this experiment was to test the effectiveness of composite filters made from citrus peels and citrus pectin along with charcoal and sand on removing heavy metal pollutants from the waters of Tar Creek. A toxicity test was also done before and after filtration using Daphnia magna. Charcoal and sand were used as filtrates to decrease the TDS and neutralize the pH of the water after filtration. Daphnia magna were used as toxicity test before and after filtration. It was hypothesized that the composite filters (citrus + sand +charcoal) will decrease the heavy metal concentration, neutralize the pH, and decrease the TDS after filtration. It was also hypothesized that a higher percentage of Daphnia magna will survive in the filtered water as compared to the unfiltered water. Water samples were collected from four different sites at Tar Creek. Each water sample went through four different citrus filters plus one control (sand + charcoal). All the citrus filters decreased the heavy metal concentration after filtration. All of the filters neutralized the pH. The citrus peel filters for Site 4 were the only filters to have a pH of 7 after filtration. Only 25% of the citrus filters decreased the TDS after filtration, while 50% of the control filters decreased the TDS after filtration. A higher percentage of Daphnia magna survived after filtration. The orange peel had the overall highest survival of Daphnia after filtration. The correlation observed before and after filtration was cadmium was most toxic to Daphnia magna."
# "The Southwest shrub Juniperus communis (Juniper Berry) has many significant medicinal value in the Native American culture that has not been proven scientifically. One of the popular uses of Juniper berries aside from its detoxifying action is its potential to repel insects. This study focuses on the development of insect repellant from its essential oil obtained through steam distillation. 50 g of fresh berries was collected and dried for 5 days and is placed in a still tank with 100 mL of water for steam distillation using the Flinn Scientific Borosilicate Lab Kit. Gather the extracted oil and dilute 70% in three separate containers to be transferred into spray bottles. Testing involved the spraying of the dilute sample into a class jar with Anopheles juidthae (common NM mosquito) and compared this to the effect of a commercial insect repellant. After testing and comparing the result, the commercial insect repellant significantly showed that it is a better insect repellant compared to the J. communis diluted essential oil. However, the essential oil has also an insect repellant potential"

# ff=summarize(text)
# print(ff)