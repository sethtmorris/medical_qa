import csv

def main():
    raw_qa_file_path = "data/mle_screening_dataset.csv"
    text_data = ""
    q_list = []
    a_list = []
    with open(raw_qa_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            qa = row[0].split(',')
            q_list.append(qa[0]+'\n')
            a_list.append(qa[-1]+'\n')

    n_qa = len(q_list)
    test_n = int(0.8 * n_qa)

    prefixed_question_file_path = "data/prefixed_test_questions.txt"
    with open(prefixed_question_file_path, 'w') as question_file:
        question_file.writelines(q_list)

    test_answers_file_path = "data/test_answers.txt"
    with open(test_answers_file_path, 'w') as answer_file:
        answer_file.writelines(a_list)

if __name__ == "__main__":
    main()
