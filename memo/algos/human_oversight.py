import sys
sys.path.insert(1, '/home/tyna/Documents/memo/')  # insert at 1, 0 is the script path (or '' in REPL)
from memo.utils.utils import load_policy_and_env
import os.path as osp
import streamlit as st

# Fetch the model
memo_path = '/home/tyna/Documents/memo/memo/data/'
exp_name = "vm-memo-run-200"


# # Load the model
# _, memo_pi = load_policy_and_env(osp.join(memo_path, exp_name, exp_name + '_s0/'), 'last', False)
#
#
# # create a prompt text for the text generation
# #prompt_text = "Python is awesome"
# # if (name != ""):
# #     # Print welcome message if the value is not empty
#
# st_name = st.text_input("Hello, my name is MEMO. What is your name? ")
#
# # st.text(st_name)
# st.text("Welcome to the MEMO playground, _____. Here are the environments we have available:")
#
# prompt_text = st.text_input(label = "Enter your prompt text...",
#             value = "Computer is beautiful")
#
# with st.spinner("AI is at Work........"):
#     pass

# st.success("AI Successfully generated the below text ")
# st.balloons()
# # print ai generated text
# print(gpt_text)
#
# st.text(gpt_text)
#

# name = input("Hello, my name is MEMO. What is your name? ")

# chosen_env = input("What environment would you like to play?")
# if (chosen_env != ""):
#     print("The last time I played in the %s environment, I learned ____ distinct behaviors." % chosen_env)
#
# demo_choice = input("Would you like to see demos? [Y/N]")
# if demo_choice.lower() == "y":
#     print("hurray!")
#
#

CHARACTER_IMAGE_PATHS = {"TONY STARK": "/images/tony-stark.png",
                         "BRUCE BANNER": "/images/bruce-banner.png",
                         "PEPPER POTTS": "/images/pepper-potts.png",
                         "NATASHA ROMANOFF": "/images/natasha.png",
                         "LOKI": "/images/loki.png",
                         "STEVE ROGERS": "/images/steve-rogers.png",
                         "THOR": "/images/thor.png",
                         "NICK FURY": "/images/nick-fury.png",
                         "PETER PARKER": "/images/peter-parker.png",
                         "JAMES RHODES": "/images/james-rhodes.png"}


# class StemCountVectorizer(CountVectorizer):
#     def build_analyzer(self):
#         analyzer = super(StemCountVectorizer, self).build_analyzer()
#
#         return lambda document: (
#             [SnowballStemmer('english', ignore_stopwords=True).stem(word) for word in analyzer(document)])
#

class Application():
    def __init__(self):
        # Fetch the model
        self.memo_path = '/home/tyna/Documents/memo/memo/data/'
        self.file_path = '/home/tyna/Documents/memo/memo/'
        self.exp_name = "vm-memo-run-200"

        # self.file_path = "/app/marvel-dialogue-nlp/front_end"

        # self.input_string = None
        # self.model = load(self.file_path + '/production_model.joblib')

        # self.model_predictions = load(self.file_path + '/production_predictions.joblib')
        # self.character_similarity = load(self.file_path + '/character_similarity.joblib')
        # self.main_characters = self.model_predictions["true character"].value_counts().index.to_numpy()

        self.prediction = None
        self.prediction_conf = None
        self.prediction_probs = None

        self.rank_table = None
        self.hierarchical_rank_table = None

        # self.recalls = pd.read_csv(self.file_path + "/production_recalls.csv")
        self.confusion_matrix = None

    def render_header(self):

        st.markdown(
            '<style>.text{font-family: "IBM Plex Mono", monospace; white-space: pre;font-size: 0.8rem; overflow-x: auto;}</style>',
            unsafe_allow_html=True)

        st.title("Multiple Experts, Multiple Objectives")
        st.text("By Florentine (Tyna) Eloundou")
        st.text("")
        st.text("")

        st.text("This page presents a deep learning project on example-based imitation learning with \n"
                "multiple experts.")

        st.markdown(
            '<p class="text">For more about the project see its <a class="text" href="https://github.com/feloundou/memo" target="_blank">GitHub repository</a>.  Feel free to contact me at\n<a class="text" href="mailto:mfe25@cornell.edu">mfe25@cornell.edu</a>.</p>',
            unsafe_allow_html=True)
        st.markdown('', unsafe_allow_html=True)

        st.text("")
        st.text("")

    def render_interactive_prediction(self):

        st.header("Welcome!")
        st.text("Hello, my Name is MEMO.")

        self.input_string = st.text_input('What is your name?', 'Hi MEMO, I am R2D2.')

        # self.prediction = self.model.predict([self.input_string])
        # self.prediction_conf = self.model.predict_proba([self.input_string]).max()
        col1, col2, col3 = st.beta_columns(3)

        st.markdown('<style>.prediction{color: red; font-size: 24px; font-weight: bold}</style>', unsafe_allow_html=True)

        # with col1:
        #     st.subheader("Prediction:")
        #     st.markdown('<p class="prediction">' + self.prediction[0].title() + '</p>', unsafe_allow_html=True)
        # with col2:
        #     st.subheader("Confidence:")
        #     st.markdown('<p class="prediction">' + "{0:.3%}".format(self.prediction_conf) + '</p>',
        #                 unsafe_allow_html=True)
        # with col3:
        #     st.image(self.file_path + CHARACTER_IMAGE_PATHS[self.prediction[0]], width=200)
        #
        # self.render_probability_table()

    # def render_probability_table(self):
    #
    #     # st.subheader("Probability Table")
    #
    #     vect = self.model.named_steps['vect']
    #     tokenizer = vect.build_tokenizer()
    #     prediction_array = tokenizer(self.input_string)
    #     prediction_array.append(self.input_string)
    #
    #     probabilities = pd.DataFrame(self.model.predict_proba(prediction_array).transpose())
    #     probabilities.columns = prediction_array
    #     probabilities.columns = [*probabilities.columns[:-1], 'Combined Probability']
    #     probabilities.insert(0, "character", self.model.classes_)
    #     probabilities.set_index('character', inplace=True)
    #     probabilities.sort_values(by=['Combined Probability'], ascending=False, inplace=True)
    #
    #     used_column_names = []
    #     column_names = probabilities.columns.to_numpy()
    #     for i in range(0, len(column_names)):
    #         while column_names[i] in used_column_names:
    #             column_names[i] = column_names[i] + " "
    #
    #         used_column_names.append(column_names[i])
    #
    #     probabilities.columns = column_names
    #
    #     self.prediction_probs = probabilities
    #
    #     # st.dataframe(self.prediction_probs.style.background_gradient(cmap=plt.cm.Reds, high=0.35))
    #
    #     st.text("The table above shows the probabilities the model predicts given a character the input.\n"
    #             "Each cell holds the probability of predicting its row's character given its column's\n"
    #             "word. In other words:")
    #     st.latex("cell(row, column)=P(character|word)")
    #     st.text("The final column represents the probability our model predicts a character given the\n"
    #             "entire input string together.  The character with the largest value in this column\n"
    #             "is our model's prediction.  One character words like 'I' and 'a' are removed because\n"
    #             "they don't provide any useful information to the model.  No other words are removed.")
    #     st.text("By clicking on the names of the columns, you can sort the table and see which\n"
    #             "character is most likely to say a word.")

    def render_about_the_model(self):
        st.header("About the Model")

        st.subheader("Implementation Details")
        st.markdown(
            '<p class="text">This project uses <a href="https://pytorch.org/" target="_blank">Pytorch</a> to implement a \n'
            'Classifier-Generator architecture. In this demo, we use examples from 5 different types of agents: \n'
            'a very risk-averse agent. \n'
            'better performance keeping all words rather than removing <a href="https://www.geeksforgeeks.org/removing-stop-words-nltk-python/" target="_blank">NLTK\'s list of stop words</a>.\n'
            'Words are stemmed using <a href="https://www.nltk.org/_modules/nltk/stem/snowball.html" target="_blank">NLTK\'s SnowballStemmer</a>.  Word counts are also transformed with\n'
            'term frequencies and inverse document frequencies using scikit-learn\'s implementation.</p>',
            unsafe_allow_html=True)
        st.markdown(
            '<p class="text">To see the code for the model, see the <a href="https://github.com/feloundou/memo/blob/master/memo/algos/run_memo_experts.py" target="_blank">Launch MEMO Training</a> here.</p>',
            unsafe_allow_html=True)

        st.subheader("Environment Details")


        st.subheader("Why PointGoal?")
        st.markdown(
            "<p class='text'>While most RL environments are generally encoded with rewards and penalties, \n"
            "Safety Gym defines this explicitly, and is flexible enough to modify an agent's abilities, \n"
            "and the environmental constraints.</p>", unsafe_allow_html=True)
        st.markdown(
            '<p class="text">For details on the expert sample creation, \n'
            'characters, see the <a href="https://github.com/feloundou/memo/blob/master/algos/run_memo_experts" target="_blank">Make Experts</a> script.</p>',
            unsafe_allow_html=True)

        # st.subheader("Other Models")
        # st.markdown(
        #     "<p class='text'>In this project, 18 different models were buit and compared.  Models 1-12 use Naive Bayes,\n"
        #     "SVM, and Random Forest classifiers in different architecture combinations and can be read\n"
        #     "about in the <a href='https://github.com/prestondunton/marvel-dialogue-nlp/tree/master/old%20models' target='_blank'>old models directory</a>. Model 13 is the Naive Bayes classifier with the best\n"
        #     "performance and presented here as the production model.  Models 14-18 are derived from\n"
        #     "model 13, but manipulated the data or larger architecture to try to achieve better\n"
        #     "results.  Model 14 is an ensemble method that trains a model for every character and can\n"
        #     "be read about in the <a href='https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/One%20vs.%20Rest%20Models.ipynb' target='_blank'>One vs. Rest Models</a> notebook.  Model 15 allows the use of movie\n"
        #     "titles and authors as features and can be read about in the <a href='https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/All%20Features%20Model.ipynb' target='_blank'>All Features Model</a> notebook.\n"
        #     "Models 16, 17, and 18 were inspired by the correlation between the number of words in a\n"
        #     "line and its correct prediction, shown in the section below.  These models attempt to\n"
        #     "train on less sparse vectors and can be read about in the <a href='https://github.com/prestondunton/marvel-dialogue-nlp/blob/master/Word%20Count%20Models.ipynb' target='_blank'>Word Count Models</a> notebook.</p>",
        #     unsafe_allow_html=True)

    def render_model_performance(self):
        st.header("Model Performance")

        st.markdown(
            '<p class="text">To see the code for these metrics, and more metrics, see the <a href="https://github.com/feloundou/memo/blob/master/utils/utils/" target="_blank">Context Output Utils</a>.</p>', unsafe_allow_html=True)

        self.render_confusion_matrix()
        self.render_recalls()
        self.render_accuracy_by_words()

        # y = self.model_predictions['true character']
        # yhat = self.model_predictions['predicted character']
        # st.subheader("Model's Balanced Accuracy: {0:.3%}".format(metrics.balanced_accuracy_score(y, yhat)))

        st.text("The model's performance depends on the diversity of samples, but is fun to explore.\n"
                "More diverse samples will lead to very distinct classification.")

    def render_confusion_matrix(self):
        # y = self.model_predictions['true character']
        # yhat = self.model_predictions['predicted character']

        st.subheader("Confusion Matrix")

        st.text("The plot below summarizes the predictions of our model.  Each cell represents the\n"
                "proportion of all of a true character's examples that are predicted as a character.")
        st.text("The diagonal elements represent examples that our model correctly predicts, as well as\n"
                "the recall for that character.")

        # conf_matrix = pd.DataFrame(metrics.confusion_matrix(y, yhat, labels=self.main_characters))
        # normalized_conf_matrix = conf_matrix.div(conf_matrix.sum(axis=1), axis=0)
        # normalized_conf_matrix.columns = pd.Series(self.main_characters, name="Predicted Character")
        # normalized_conf_matrix.index = pd.Series(self.main_characters, name="True Character")

        # self.confusion_matrix = normalized_conf_matrix

        # fig = plt.figure(figsize=(2, 2))
        # fig, ax = plt.subplots()
        # ax = sns.heatmap(normalized_conf_matrix, annot=True, fmt='.2f', cmap=plt.cm.Reds)
        #
        # st.pyplot(fig)

    def render_recalls(self):
        st.subheader("Accuracy by Context")
        st.text("Given a line we'd like to predict from a given character, here's how often we can expect\n"
                "our model to be correct.")

        # self.recalls.set_index("Unnamed: 0", drop=True, inplace=True)
        # st.dataframe(self.recalls)

    def render_accuracy_by_words(self):

        st.subheader("Performance Vs. Contexts")

        st.text("Does training with more types of experts affect classification performance?")

        def abline(intercept, slope, col):
            """Plot a line from slope and intercept"""
            # axes = plt.gca()
            # x_vals = np.array(axes.get_xlim())
            # y_vals = intercept + slope * x_vals
            # plt.plot(x_vals, y_vals, '-', color=col)

        # regression_data = self.model_predictions.copy(deep=True)

        # regression_data['words'] = regression_data['line'].str.split(" ").str.len()
        # regression_data['correct_prediction'] = (
        #             regression_data['true character'] == regression_data['predicted character']).astype('int64')

        # reg_model = smf.ols('regression_data["correct_prediction"] ~ regression_data["words"]',
        #                     data=regression_data).fit()

        # fig = plt.figure(figsize=(2, 2))
        # fig, ax = plt.subplots()
        #
        # ax = plt.scatter(x=regression_data['words'].to_numpy(),
        #                  y=regression_data['correct_prediction'].to_numpy(),
        #                  color='black')

        # abline(reg_model.params[0], reg_model.params[1], 'red')
        #
        # conf_pred_intervals = reg_model.get_prediction(regression_data['words']).summary_frame()

        # plt.fill_between(regression_data['words'].to_numpy(), conf_pred_intervals['mean_ci_lower'],
        #                  conf_pred_intervals['mean_ci_upper'], alpha=0.3, color='red')
        #
        # plt.grid()
        # plt.ylim(-0.1, 1.2)
        # plt.title('Performance vs. Length of Words (with 95% CI Band)')
        # plt.xlabel('words')
        # plt.ylabel('accuracy')
        #
        # st.pyplot(fig)



    def render_context_insights(self):
        st.header("Insights")


        self.render_character_similarity()
        # self.render_context_development()

    def render_character_similarity(self):
        st.subheader("Context Similarity and the SSPD metric")
        st.text("The SSPD metric used here is the ...")
        st.text("It should be noted that we calculate the SSPD metric based on actions taken rather than the\n"
                "trajectory because ...")

        # fig = plt.figure(figsize=(2, 2))
        # fig, ax = plt.subplots()
        # ax = sns.heatmap(self.character_similarity, annot=True, fmt='.2f', cmap=plt.cm.Reds)
        #
        # st.pyplot(fig)

        st.markdown('<p class="text" style="font-weight: bold;">What do these SSPD scores mean?</p>',
                    unsafe_allow_html=True)
        st.markdown(
            '<p class="text"> Because ...</p>', unsafe_allow_html=True)

        st.markdown(
            '<p class="text" style="font-weight: bold;">Why does Rose expert present as more unique among the experts?</p>',
            unsafe_allow_html=True)

        st.text("The goal-seeking agent ")

        st.text("Another agent ...")

        st.text("Finally... ")

        st.markdown(
            "<p class=\"text\">The probable answer is that Rose is so unique for all of the reasons mentioned\n"
            "completed and added to the dataset, it would be interesting to see how that affects this\n"
            "similarity.</p>", unsafe_allow_html=True)


    def render_model_predictions(self):
        st.header("Model Predictions")

        # table = self.model_predictions
        # table['correct prediction'] = table['true character'] == table['predicted character']
        # table['correct prediction'] = table['correct prediction'].replace({0: 'No', 1: "Yes"})

        # true_character_filter = st.multiselect("True Character", list(self.main_characters), ["PETER PARKER"])
        # pred_character_filter = st.multiselect("Predicted Character", list(self.main_characters), ["PETER PARKER"])
        # movie_filter = st.multiselect("Movie", list(self.model_predictions['movie'].unique()),
        #                               ["Captain America: Civil War", 'Avengers: Endgame'])

        # if len(true_character_filter) == 0:
        #     true_character_filter = self.main_characters
        # if len(pred_character_filter) == 0:
        #     pred_character_filter = self.main_characters
        # if len(movie_filter) == 0:
        #     movie_filter = self.model_predictions['movie'].unique()

        # st.table(table[table['true character'].isin(true_character_filter) &
        #                table['predicted character'].isin(pred_character_filter) &
        #                table['movie'].isin(movie_filter)])

    def render_app(self):
        # st.set_page_config(page_title='Marvel Dialogue Classification', layout='centered', \
        #                    initial_sidebar_state='auto', page_icon=self.file_path + "/images/marvel-favicon.png")

        self.render_header()

        st.image(osp.join(self.file_path + "/images/horizontal_line.png"), use_column_width=True)
        self.render_interactive_prediction()

        st.text(" ")
        st.text(" ")
        st.text(" ")

        st.image(self.file_path + "/images/horizontal_line.png", use_column_width=True)
        self.render_about_the_model()

        st.text(" ")
        st.text(" ")
        st.text(" ")

        st.image(self.file_path + "/images/horizontal_line.png", use_column_width=True)
        self.render_model_performance()

        st.text(" ")
        st.text(" ")
        st.text(" ")

        st.image(self.file_path + "/images/horizontal_line.png", use_column_width=True)
        self.render_context_insights()

        st.text(" ")
        st.text(" ")
        st.text(" ")

        st.image(self.file_path + "/images/horizontal_line.png", use_column_width=True)
        self.render_model_predictions()


app = Application()
app.render_app()
