import re
import math
import collections
import nltk


class Spell_Checker:
    """The class implements a context sensitive spell checker. The corrections
        are done in the Noisy Channel framework, based on a language model and
        an error distribution model.
    """

    def __init__(self, lm=None):
        """Initializing a spell checker object with a language model as an
        instance variable. The language model supports the evaluate()
        and the get_model() functions as defined in assignment #1.

        Args:
            lm: a language model object. Defaults to None
        """
        self.lm = lm
        self.error_tables = None

    def build_model(self, text, n=3):
        """Returns a language model object built on the specified text. The language
            model supports evaluate() and the get_model() functions as defined
            in assignment #1.

            Args:
                text (str): the text to construct the model from.
                n (int): the order of the n-gram model (defaults to 3).

            Returns:
                A language model object
        """
        lm = Ngram_Language_Model(n=n)
        normalized_text = lm.normalize_text(text)
        lm.build_model(normalized_text)
        self.add_language_model(lm)
        return lm

    def add_language_model(self, lm):
        """Adds the specified language model as an instance variable.
            (Replaces an older LM dictionary if set)

            Args:
                lm: a language model object
        """
        self.lm = lm

    def learn_error_tables(self, errors_file):
        """Returns a nested dictionary {str:dict} where str is in: <'deletion', 'insertion', 'transposition',
        'substitution'> and the inner dict {str: int} represents the confusion matrix of the specific errors,
        where str is a string of two characters matching the row and column "indixes" in the relevant confusion
        matrix and the int is the observed count of such an error (computed from the specified errors file).

        Notes:
            1. records such as "didnt      didn't" that contains the sign `'` were addressed in the
            dictionary, as all other characters.  the sign `'`is prevalent in English, especially as in
            deletion/transposition errors.
            2. records such as "eventhough      even though." were addressed in the dictionary. a deletion of
            " " is very common error, and should be addressed as any other character
            3.Normalization: lower casing was performed, but no other operations such as punctuation padding- to
            prevent unwanted errors in the input words.

        Args:
            errors_file (str): full path to the errors file. File format, TSV:
                                <error>    <correct>


            Returns:
                A dictionary of confusion "matrices" by error type (dict).


        """
        error_tables = dict({"insertion": {}, "deletion": {}, "substitution": {}, "transposition": {}})

        # load the errors file
        f = open(errors_file, "r")
        errors_text = f.read()
        errors_text = errors_text.lower()

        # get all misspelled words
        misspelled_rgx = "\S+\t"
        misspelled_words = re.findall(misspelled_rgx, errors_text)
        misspelled_words = [var.replace("\t", "") for var in misspelled_words]

        # get all correct words
        correct_rgx = r"\t\S+\s+\S+\n|\t\S+|\s\s\S+"
        correct_words = re.findall(correct_rgx, errors_text)
        correct_words = [var.replace("\t", "") for var in correct_words]
        correct_words = [var.replace("\n", "") for var in correct_words]

        for misspelled_w, correct_w in zip(misspelled_words, correct_words):
            # get all matching error types
            error_types = self.get_error_types(misspelled_w, correct_w)
            for error_type in error_types:  # count occurrences insert each error type to the error_tables
                error_str = self.get_error(error_type, misspelled_w, correct_w)
                occur_num = error_tables[error_type].setdefault(error_str, 0)
                error_tables[error_type].update({error_str: occur_num + 1})

        self.add_error_tables(error_tables)

        return error_tables

    def get_error(self, error_type, misspelled_w, correct_w):
        """
        Get a two character string, represents the error found in the misspelled word.

        Args:
            error_type (str): insertion, deletion, transposition or substitution
            misspelled_w (str): misspelled word
            correct_w (str): correct word
        Return:
            (str):  a two character string represents the error found
        """
        if error_type == "insertion":
            return self.get_insertion_misspelling_chars(misspelled_w, correct_w)

        elif error_type == "deletion":
            return self.get_deletion_misspelling_chars(misspelled_w, correct_w)

        elif error_type == "transposition":
            return self.get_transposition_misspelling_chars(misspelled_w, correct_w)

        elif error_type == "substitution":
            return self.get_substitution_misspelling_chars(misspelled_w, correct_w)

    def add_error_tables(self, error_tables):
        """ Adds the specified dictionary of error tables as an instance variable.
            (Replaces an older value dictionary if set)

            Args:
                error_tables (dict): a dictionary of error tables in the format
                returned by  learn_error_tables()
        """
        self.error_tables = error_tables

    def evaluate(self, text):
        """Returns the log-likelihod of the specified text given the language
            model in use. Smoothing is applied on texts containing OOV words

           Args:
               text (str): Text to evaluate.

           Returns:
               Float. The float should reflect the (log) probability.
        """
        return self.lm.evaluate(text)

    def spell_check(self, text, alpha):
        """ Returns the most probable fix for the specified text. Use a simple
            noisy channel model if the number of tokens in the specified text is
            smaller than the length (n) of the language model.

            Args:
                text (str): the text to spell check.
                alpha (float): the probability of keeping a lexical word as is.

            Return:
                A modified string (or a copy of the original if no corrections are made.)
        """
        copied_text = (text + '.')[:-1]
        all_words = copied_text.split(" ")
        correct_words = []

        for word in all_words:
            try:
                all_candidates = self.edits1(word)

                # ignore punctuation and numbers
                if word in "!#$%&'()*+, -./:;<=>?@[\]^_`{|}~" or word.isnumeric():
                    correct_words.append(word)
                    continue

                # if word is not in the dictionary
                # or the number of tokens in the input text is smaller than the length (n) of the lm
                if (self.lm.unigram_dict.get(word, None) is None) or (len(all_words) < self.lm.n):
                    # Use a simple noisy channel model
                    c_word = self.get_next_word(all_candidates, word)

                else:  # 'word' is a dictionary word
                    all_candidates = self.edits1(word)
                    replacement_dict = {}
                    for edit_type in all_candidates.keys():
                        edit1_options = all_candidates.get(edit_type)
                        for candidate in edit1_options:
                            if word == candidate:  # candidate is the same word
                                # if x = w
                                replacement_dict.update({candidate: alpha})
                            else:  # if x ∈ C(x)
                                c_x_size = sum(len(v) for v in all_candidates.values())
                                p_candidate = (1 - alpha) / c_x_size
                                error = self.get_error(edit_type, word, candidate)
                                p_x_w = self.compute_noisy_channel(edit_type, error)
                                log_prob = math.log(p_candidate) + math.log(p_x_w)
                                replacement_dict.update({candidate: log_prob})
                    c_word = max(replacement_dict, key=replacement_dict.get) if len(
                        replacement_dict.items()) > 0 else word
            except:
                c_word = word
            correct_words.append(c_word)
        return " ".join(correct_words)

    def get_next_word(self, replacement_options, word):
        """
        Uses a simple noisy channel model to find the most probable word

        Args:
            replacement_options (dict): all the word's candidates
            word (str): word to replace

        Return:
            (str) the most probable word
        """
        replacement_dict = {}
        for edit_type in replacement_options.keys():
            edit1_options = replacement_options.get(edit_type)
            for edit in edit1_options:
                if self.lm.unigram_dict.get(edit, None) is None:
                    continue
                error = self.get_error(edit_type, word, edit)
                if error is None:
                    return edit
                p_x_w = self.compute_noisy_channel(edit_type, error)
                pw = self.get_prior(edit)
                log_prob = math.log(p_x_w) + math.log(pw)
                replacement_dict.update({edit: log_prob})

        c_word = max(replacement_dict, key=replacement_dict.get) if len(
            replacement_dict.items()) > 0 else word
        return c_word

    def edits1(self, word):
        """
        Returns all dits that are one edit away from `word`.

        Args:
            word (str): original word to calculate edits from

        Return:
            (dict): {str:list} of all candidates where str is in: <'deletion', 'insertion', 'transposition',
        'substitution'> , and list is all possible candidates.

        Notes:
            taken from: http://norvig.com/spell-correct.html
        """

        unigrams_dict = self.lm.unigram_dict

        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

        inserts = [L + R[1:] for L, R in splits if R]
        inserts = [ins for ins in inserts if unigrams_dict.get(ins, None) is not None]

        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        transposes = [trans for trans in transposes if unigrams_dict.get(trans, None) is not None]

        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        replaces = [rep for rep in replaces if unigrams_dict.get(rep, None) is not None]

        deletes = [L + c + R for L, R in splits for c in letters]
        deletes = [delete for delete in deletes if unigrams_dict.get(delete, None) is not None]

        return dict({"deletion": deletes, "transposition": transposes, "substitution": replaces, "insertion": inserts})

    def get_error_types(self, misspelled_w, correct_w):
        """
        Find the all matching error types of misspelled and correct word

        Args:
            misspelled_w (str): misspelled word
            correct_w (str): correct word

        Return:
            (list): the error types (insertion/deletion/transposition/substitution)

        """
        res = []
        if len(misspelled_w) > len(correct_w):
            res.append("insertion")
        if len(misspelled_w) < len(correct_w):
            res.append("deletion")
        if len(misspelled_w) == len(correct_w) and sorted(misspelled_w) == sorted(correct_w):
            res.append("transposition")
        if len(misspelled_w) == len(correct_w) and (sorted(misspelled_w) != sorted(correct_w)):
            res.append("substitution")
        return res

    def get_insertion_misspelling_chars(self, misspelled_w, correct_w):  # 'xy' , for insertion of a 'y' after an 'x'
        """
        Get a string of 2 characters, represents the insertion error

        Args:
            misspelled_w (str): misspelled word
            correct_w (str): correct word

        Return:
            (str) the error
        """
        for i, c in enumerate(correct_w):
            if c != misspelled_w[i]:
                return misspelled_w[i - 1] + misspelled_w[i]
        return misspelled_w[i] + misspelled_w[i + 1]

    def get_deletion_misspelling_chars(self, misspelled_w, correct_w):  # 'xy', for deletion of a 'y' after an 'x'
        """
        Get a string of 2 characters, represents the deletion error

        Args:
            misspelled_w (str): misspelled word
            correct_w (str): correct word

        Return:
            (str) the error
        """
        for i, m in enumerate(misspelled_w):
            if m != correct_w[i]:
                return misspelled_w[i - 1] + correct_w[i]
        return correct_w[i] + correct_w[i + 1]  # in case the last char was deleted

    def get_transposition_misspelling_chars(self, misspelled_w,
                                            correct_w):  # 'xy' , indicates the characters that are transposed.
        """
        Get a string of 2 characters, represents the transposition error

        Args:
            misspelled_w (str): misspelled word
            correct_w (str): correct word

        Return:
            (str) the error
        """
        for i, m in enumerate(misspelled_w):
            if correct_w[i] != m:
                return misspelled_w[i + 1] + m

    def get_substitution_misspelling_chars(self, misspelled_w,
                                           correct_w):  # 'xy' , for substitution of 'x' (incorrect) by a 'y'
        """
        Get a string of 2 characters, represents the substitution error

        Args:
            misspelled_w (str): misspelled word
            correct_w (str): correct word

        Return:
            (str) the error
        """
        for c, m in zip(correct_w, misspelled_w):
            if c != m:
                return m + c

    def get_counts(self, str_to_count):
        """
        Counts the number of error's occurrences in the language model.

        Args:
            str_to_count (str): the error to count

        Return:
            (float): number of error's occurrences in the language model.
            if there are any occurrences, return 0.0001
        """
        counter = 0
        ngram_dict = self.lm.get_model()
        for key in ngram_dict.keys():
            counter += key.count(str_to_count) * ngram_dict.get(key, 0)
        return counter if counter > 0 else 0.0001

    def compute_noisy_channel(self, edit_type, error):
        """
        Computes thr noisy channel formula.

        Args:
            edit_type (str): (insertion/deletion/transposition/substitution)
            error (str): a string of 2 characters, represents the error

        Return:
            (float): p(x|w)
        """
        count_error = self.error_tables.get(edit_type).get(error, 0)
        count_error = 0.0001 if count_error == 0 else count_error

        if edit_type == "deletion":
            # del[x,y] = the number of times that the characters xy (in the correct word) were typed as x
            return count_error / self.get_counts(error)

        elif edit_type == "insertion":
            # ins[x,y] = the number of times that x was typed as xy
            return count_error / self.get_counts(error[0])

        elif edit_type == "substitution":
            # sub[x,y] = the number of times that y was typed as x
            return count_error / self.get_counts(error[1])

        elif edit_type == "transposition":
            # trans[x,y] = the number of times that xy was typed as yx
            return count_error / self.get_counts(error)

    def get_prior(self, word):
        """
        Calculates the prior probability

        Args:
            word (str): word to calculate its probability

        Return:
            (float): the prior probability
        """
        vocabulary_size = sum(self.lm.unigram_dict.values())
        occur_num = self.lm.count_occure(word)
        return occur_num / vocabulary_size


class Ngram_Language_Model:
    """The class implements a Markov Language Model that learns a language model
        from a given text.
        It supports language generation and the evaluation of a given string.
        The class can be applied on both word level and character level.
    """

    def __init__(self, n=3, chars=False):
        """Initializing a language model object.
        Args:
            n (int): the length of the markov unit (the n of the n-gram). Defaults to 3.
            chars (bool): True iff the model consists of ngrams of characters rather then word tokens.
                          Defaults to False.
        """
        self.n = n
        self.model_dict = collections.defaultdict(
            int)  # a dictionary of the form {ngram:count}, holding counts of all ngrams in the specified text.
        self.chars = chars
        self.unigram_dict = collections.defaultdict(int)
        self.split_by = "" if self.chars else " "

    def split_to_unigrams(self, text):
        """
        Splits a text into a list of tokens
            Args:
                text (str): the text to split.
            Return:
                (list): tokens, splitted by "" if the model consists of ngrams of characters rather then word tokens,
                else by " ".
        """
        return list(text) if self.chars else text.split(" ")

    def split_to_n_grams(self, text):
        """
        Splits the input text into n-grams
            Args:
                text (str): the text to split.
            Return:
                (list): n-grams list
        """
        n_grams = []
        splitted_text = self.split_to_unigrams(text)
        if len(splitted_text) < self.n:
            return splitted_text
        for i in range(len(splitted_text) - (self.n - 1)):
            n_gram = self.split_by.join(splitted_text[i:i + self.n])
            n_grams.append(n_gram)
        return n_grams

    def build_model(self, text):
        """populates the instance variable model_dict.

            Args:
                text (str): the text to construct the model from.
        """
        n_grams_list = nltk.ngrams(nltk.word_tokenize(text), self.n)
        unigrams = nltk.ngrams(nltk.word_tokenize(text), 1)

        # build the n-gram dictionary
        for n_gram in n_grams_list:
            n_gram_str = self.split_by.join(n_gram)
            occur_num = self.model_dict.setdefault(n_gram_str, 0)
            self.model_dict.update({n_gram_str: occur_num + 1})

        # build the unigram dictionary
        for unigram in unigrams:
            u_gram_str = self.split_by.join(unigram)
            occur_num = self.unigram_dict.setdefault(u_gram_str, 0)
            self.unigram_dict.update({u_gram_str: occur_num + 1})

    def get_model_dictionary(self):
        """Returns the dictionary class object
        """
        return self.model_dict

    def get_model(self):
        """Returns the dictionary class object
        """
        return self.model_dict

    def get_model_window_size(self):
        """Returning the size of the context window (the n in "n-gram")
        """
        return self.n

    def get_markov_n_minus_dict(self, ngram):
        """
        finds all the ngram's matching keys in the model
            Args:
                ngram (str): the ngram to find
            Return:
                (defaultdict): all options found in the model

        """
        markov_options = {}
        for key in self.model_dict.keys():
            if key.startswith(ngram):
                mo_key = key if key == "" else self.split_to_unigrams(key)[self.n - 1]
                markov_options[mo_key] = self.model_dict.get(key)
        return markov_options

    # def sample_context(self):
    #     """
    #     Samples a new context from the model's distribution
    #         Return:
    #             (str): a sampled context
    #     """
    #     return random.choices(list(self.model_dict.keys()), weights=list(self.model_dict.values()))[0]

    # def is_exhausted_context(self, context):
    #     """
    #     Checks if the context is exhausted or not
    #         Args:
    #             (str): the context
    #         Return:
    #             (bool): True if the context is exhausted, False otherwise.
    #     """
    #     return len(self.get_markov_n_minus_dict(context)) == 0

    def count_occure(self, word):
        """
        Returns the number of word occurrences in the model
            Args:
                (str) ngram or a part of it
            Return:
                (int) num of occurrences
        """
        counter = 0
        for key in self.model_dict.keys():
            if word in key:
                counter += self.model_dict.get(key, 0)
        return counter

    def evaluate(self, text):
        """Returns the log-likelihood of the specified text to be a product of the model.
           Laplace smoothing should be applied if necessary.

           Args:
               text (str): Text to evaluate.

           Returns:
               Float. The float should reflect the (log) probability.
        """
        probs = []
        n_grams_list = self.split_to_n_grams(text)
        unigrams_size = sum(self.unigram_dict.values())  # number of unigrams in total
        ngram_size = sum(self.model_dict.values())

        first_ngram = n_grams_list[0]
        first_unigrams = [self.split_by.join(self.split_to_unigrams(first_ngram)[0:i]) for i in range(1, self.n)]

        for i, ngram in enumerate(first_unigrams):
            _ngram = ""
            _occur = self.count_occure(ngram)
            ngram_split = ngram.split(self.split_by)
            denominator = ngram_size if len(ngram_split) == 1 else self.count_occure(
                self.split_by.join(ngram_split[:i]))
            _prob = _occur / denominator if _occur > 0 else self.smooth(ngram)
            log_prob = math.log(_prob)
            probs.append(log_prob)

        for ngram in n_grams_list:
            n_gram_count = self.model_dict.get(ngram)
            partition = ngram[:self.n - 1] if self.chars else ngram.rpartition(self.split_by)[0]
            n_minus_count = sum(self.get_markov_n_minus_dict(partition).values())
            ngram_prob = self.smooth(ngram) if n_gram_count is None else n_gram_count / n_minus_count
            log_prob = math.log(ngram_prob)
            probs.append(log_prob)
        return sum(probs)

    def smooth(self, ngram):
        """Returns the smoothed (Laplace) probability of the specified ngram.
            Args:
                ngram (str): the ngram to have it's probability smoothed

            Returns:
                float. The smoothed probability.
        """

        n_grams_size = sum(self.model_dict.values())  # total num of tokens in the dictionary
        v = sum(self.unigram_dict.values())  # all words in corpus
        return 1 / (n_grams_size + v)

    def normalize_text(self, text):
        """Returns a normalized version of the specified string.
            Performs the following operations:
            * Lower casing
            * Padding punctuation with white spaces
            * Expanding contractions - for example: ain't --> am not

          Args:
            text (str): the text to normalize

          Returns:
            string. the normalized text.
        """

        def expand_contractions(s):
            contractions_dict = {
                "ain't": "am not",
                "aren't": "are not",
                "can't": "cannot",
                "can't've": "cannot have",
                "'cause": "because",
                "could've": "could have",
                "couldn't": "could not",
                "couldn't've": "could not have",
                "didn't": "did not",
                "doesn't": "does not",
                "don't": "do not",
                "hadn't": "had not",
                "hadn't've": "had not have",
                "hasn't": "has not",
                "haven't": "have not",
                "he'd": "he would",
                "he'd've": "he would have",
                "he'll": "he will",
                "he'll've": "he will have",
                "he's": "he is",
                "how'd": "how did",
                "how'd'y": "how do you",
                "how'll": "how will",
                "how's": "how is",
                "I'd": "I would",
                "I'd've": "I would have",
                "I'll": "I will",
                "I'll've": "I will have",
                "I'm": "I am",
                "I've": "I have",
                "isn't": "is not",
                "it'd": "it had",
                "it'd've": "it would have",
                "it'll": "it will",
                "it'll've": "it will have",
                "it's": "it is",
                "let's": "let us",
                "ma'am": "madam",
                "mayn't": "may not",
                "might've": "might have",
                "mightn't": "might not",
                "mightn't've": "might not have",
                "must've": "must have",
                "mustn't": "must not",
                "mustn't've": "must not have",
                "needn't": "need not",
                "needn't've": "need not have",
                "o'clock": "of the clock",
                "oughtn't": "ought not",
                "oughtn't've": "ought not have",
                "shan't": "shall not",
                "sha'n't": "shall not",
                "shan't've": "shall not have",
                "she'd": "she would",
                "she'd've": "she would have",
                "she'll": "she will",
                "she'll've": "she will have",
                "she's": "she is",
                "should've": "should have",
                "shouldn't": "should not",
                "shouldn't've": "should not have",
                "so've": "so have",
                "so's": "so is",
                "that'd": "that would",
                "that'd've": "that would have",
                "that's": "that is",
                "there'd": "there had",
                "there'd've": "there would have",
                "there's": "there is",
                "they'd": "they would",
                "they'd've": "they would have",
                "they'll": "they will",
                "they'll've": "they will have",
                "they're": "they are",
                "they've": "they have",
                "to've": "to have",
                "wasn't": "was not",
                "we'd": "we had",
                "we'd've": "we would have",
                "we'll": "we will",
                "we'll've": "we will have",
                "we're": "we are",
                "we've": "we have",
                "weren't": "were not",
                "what'll": "what will",
                "what'll've": "what will have",
                "what're": "what are",
                "what's": "what is",
                "what've": "what have",
                "when's": "when is",
                "when've": "when have",
                "where'd": "where did",
                "where's": "where is",
                "where've": "where have",
                "who'll": "who will",
                "who'll've": "who will have",
                "who's": "who is",
                "who've": "who have",
                "why's": "why is",
                "why've": "why have",
                "will've": "will have",
                "won't": "will not",
                "won't've": "will not have",
                "would've": "would have",
                "wouldn't": "would not",
                "wouldn't've": "would not have",
                "y'all": "you all",
                "y'alls": "you alls",
                "y'all'd": "you all would",
                "y'all'd've": "you all would have",
                "y'all're": "you all are",
                "y'all've": "you all have",
                "you'd": "you had",
                "you'd've": "you would have",
                "you'll": "you you will",
                "you'll've": "you you will have",
                "you're": "you are",
                "you've": "you have"
            }
            contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

            def replace(match):
                return contractions_dict[match.group(0)]

            return contractions_re.sub(replace, s)

        characters_to_pad = r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*"

        # case conversion
        lower_case_txt = text.lower()

        # extend contractions
        extended_txt = expand_contractions(lower_case_txt)

        # pad punctuation
        padded_txt = re.sub(characters_to_pad, r"\1 ", extended_txt)
        trimmed_padded = padded_txt.rstrip()

        return trimmed_padded


def normalize_text(text):
    """Returns a normalized version of the specified string.
        Performs the following operations:
        * Lower casing
        * Padding punctuation with white spaces
        * Expanding contractions - for example: ain't --> am not

      Args:
        text (str): the text to normalize

      Returns:
        string. the normalized text.
    """

    def expand_contractions(s):
        contractions_dict = {
            "ain't": "am not",
            "aren't": "are not",
            "can't": "cannot",
            "can't've": "cannot have",
            "'cause": "because",
            "could've": "could have",
            "couldn't": "could not",
            "couldn't've": "could not have",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hadn't've": "had not have",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'd've": "he would have",
            "he'll": "he will",
            "he'll've": "he will have",
            "he's": "he is",
            "how'd": "how did",
            "how'd'y": "how do you",
            "how'll": "how will",
            "how's": "how is",
            "I'd": "I would",
            "I'd've": "I would have",
            "I'll": "I will",
            "I'll've": "I will have",
            "I'm": "I am",
            "I've": "I have",
            "isn't": "is not",
            "it'd": "it had",
            "it'd've": "it would have",
            "it'll": "it will",
            "it'll've": "it will have",
            "it's": "it is",
            "let's": "let us",
            "ma'am": "madam",
            "mayn't": "may not",
            "might've": "might have",
            "mightn't": "might not",
            "mightn't've": "might not have",
            "must've": "must have",
            "mustn't": "must not",
            "mustn't've": "must not have",
            "needn't": "need not",
            "needn't've": "need not have",
            "o'clock": "of the clock",
            "oughtn't": "ought not",
            "oughtn't've": "ought not have",
            "shan't": "shall not",
            "sha'n't": "shall not",
            "shan't've": "shall not have",
            "she'd": "she would",
            "she'd've": "she would have",
            "she'll": "she will",
            "she'll've": "she will have",
            "she's": "she is",
            "should've": "should have",
            "shouldn't": "should not",
            "shouldn't've": "should not have",
            "so've": "so have",
            "so's": "so is",
            "that'd": "that would",
            "that'd've": "that would have",
            "that's": "that is",
            "there'd": "there had",
            "there'd've": "there would have",
            "there's": "there is",
            "they'd": "they would",
            "they'd've": "they would have",
            "they'll": "they will",
            "they'll've": "they will have",
            "they're": "they are",
            "they've": "they have",
            "to've": "to have",
            "wasn't": "was not",
            "we'd": "we had",
            "we'd've": "we would have",
            "we'll": "we will",
            "we'll've": "we will have",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what'll": "what will",
            "what'll've": "what will have",
            "what're": "what are",
            "what's": "what is",
            "what've": "what have",
            "when's": "when is",
            "when've": "when have",
            "where'd": "where did",
            "where's": "where is",
            "where've": "where have",
            "who'll": "who will",
            "who'll've": "who will have",
            "who's": "who is",
            "who've": "who have",
            "why's": "why is",
            "why've": "why have",
            "will've": "will have",
            "won't": "will not",
            "won't've": "will not have",
            "would've": "would have",
            "wouldn't": "would not",
            "wouldn't've": "would not have",
            "y'all": "you all",
            "y'alls": "you alls",
            "y'all'd": "you all would",
            "y'all'd've": "you all would have",
            "y'all're": "you all are",
            "y'all've": "you all have",
            "you'd": "you had",
            "you'd've": "you would have",
            "you'll": "you you will",
            "you'll've": "you you will have",
            "you're": "you are",
            "you've": "you have"
        }
        contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

        def replace(match):
            return contractions_dict[match.group(0)]

        return contractions_re.sub(replace, s)

    characters_to_pad = r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*"

    # case conversion
    lower_case_txt = text.lower()

    # extend contractions
    extended_txt = expand_contractions(lower_case_txt)

    # pad punctuation
    padded_txt = re.sub(characters_to_pad, r"\1 ", extended_txt)
    trimmed_padded = padded_txt.rstrip()

    return trimmed_padded


def who_am_i():  # this is not a class method
    """Returns a dictionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Amit Shakarchy', 'id': '313278889', 'email': 'shakarch@post.bgu.ac.il'}
