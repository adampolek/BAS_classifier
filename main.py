import pickle
import pandas as pd
from absl import app, flags
from absl.flags import FLAGS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import json

flags.DEFINE_enum('classifier', 'net', ['net', 'tree', 'knn'], 'Classifier name (net - neural network, tree, knn)')
flags.DEFINE_string('input',
                    '{"Gender": ["female"], '
                    '"Glucose":[88], '
                    '"BloodPressure":[62], '
                    '"Insulin":[0], '
                    '"BMI":[28.6], '
                    '"Age":[32]}',
                    'Input of data (in json format)')
flags.DEFINE_boolean('save', True, 'Save input for future training')
flags.DEFINE_enum('job', 'test', ['train', 'test'], 'Choose to train classifier or train on it')

gender_dictionary = {'male': 0, 'female': 1, 'none': 2}
reversed_gender_dictionary = {0: "male", 1: "female", 2: "none"}


def test_method():
    data = json.loads(FLAGS.input)
    df = pd.DataFrame.from_dict(data)
    df['Gender'] = df['Gender'].map(gender_dictionary)
    with open(f"{FLAGS.classifier}_model.pkl", 'rb') as file:
        model = pickle.load(file)
    prediction = model.predict(df)
    if FLAGS.save:
        main_csv = pd.read_csv("BAS.csv")
        df['Outcome'] = prediction
        df['Gender'] = df['Gender'].map(reversed_gender_dictionary)
        main_csv = main_csv.append(df, ignore_index=True)
        main_csv.to_csv("BAS.csv", index=False)
        print(main_csv)
    print(prediction)


def train_method():
    df = pd.read_csv("BAS.csv")
    df['Gender'] = df['Gender'].map(gender_dictionary)
    y = df['Outcome']
    x = df.drop("Outcome", axis=1)
    classifier = None

    if FLAGS.classifier == 'net':
        classifier = MLPClassifier(max_iter=500, random_state=1)
    elif FLAGS.classifier == 'tree':
        classifier = DecisionTreeClassifier(random_state=1)
    elif FLAGS.classifier == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(x, y)
    with open(f"{FLAGS.classifier}_model.pkl", 'wb') as file:
        pickle.dump(classifier, file)


def main(_argv):
    if FLAGS.job == 'test':
        test_method()
    elif FLAGS.job == 'train':
        train_method()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
