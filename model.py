import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_model(sananga_name, melonidas_name, cidoso_name, n_train):
    sananga_count = 0
    melonidas_count = 0
    cidoso_count = 0

    people_names = [sananga_name, melonidas_name, cidoso_name]

    model = LogisticRegression()

    for i in range(n_train):
        while True:
            X = np.zeros((1, 3))
            X[0, 0] = 1
            X[0, 1:] = np.random.choice([0, 1], size=2)
            if np.sum(X[0, 1:]) < 2:
                break

        y = np.random.choice(people_names)

        if y == sananga_name:
            sananga_count += 1
        elif y == melonidas_name:
            melonidas_count += 1
        else:
            cidoso_count += 1

        model.fit(X, y)

        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)

        print(f"Iteração {i+1}: {y} foi a casa - Acurácia: {acc:.2f}")

    # Retornar resultado de cada pessoa
    return [
        (sananga_name, sananga_count, model.score(np.array([[1, 0, 0]]), [sananga_name])),
        (melonidas_name, melonidas_count, model.score(np.array([[0, 1, 0]]), [melonidas_name])),
        (cidoso_name, cidoso_count, model.score(np.array([[0, 0, 1]]), [cidoso_name]))
    ]

import random
import matplotlib.pyplot as plt

SANANGA = 1
MELONIDAS = 2
CIDOSO = 3

visit_counts = {SANANGA: 0, MELONIDAS: 0, CIDOSO: 0}
accuracies = []

for i in range(200):
    sananga_visited = True

    # Melônidas e Cidoso decidem aleatoriamente quem visita a casa
    if random.randint(0, 1):
        melonidas_visited = True
        cidoso_visited = False
    else:
        melonidas_visited = False
        cidoso_visited = True

    # Verifica se o modelo acertou ou errou
    if sananga_visited and not melonidas_visited and not cidoso_visited:
        accuracies.append(1)
    else:
        accuracies.append(0)

    visit_counts[SANANGA] += 1
    if melonidas_visited:
        visit_counts[MELONIDAS] += 1
    elif cidoso_visited:
        visit_counts[CIDOSO] += 1


print(f"Resultado final:\nSananga: {visit_counts[SANANGA]} visitas\nMelônidas: {visit_counts[MELONIDAS]} visitas\nCidoso: {visit_counts[CIDOSO]} visitas\nAcurácia média: {sum(accuracies)/len(accuracies):.2f}")

names = ['Sananga', 'Melônidas', 'Cidoso']
values = [visit_counts[SANANGA], visit_counts[MELONIDAS], visit_counts[CIDOSO]]
diff_counts = [visit_counts[SANANGA]-visit_counts[MELONIDAS], visit_counts[MELONIDAS]-visit_counts[CIDOSO], visit_counts[CIDOSO]-visit_counts[SANANGA]]
colors = ['blue', 'blue', 'blue', 'red', 'red', 'red']
plt.bar(names, values, color=colors)
plt.bar(names, diff_counts, color=colors)
plt.title('Estatísticas de visitas à casa')
plt.show()
