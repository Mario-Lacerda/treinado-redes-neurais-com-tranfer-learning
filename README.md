# Desafio Dio - Treinamento de Redes Neurais com Transfer Learning



**Treinamento de Redes Neurais com Transfer Learning: Um Projeto Abrangente com Códigos**

### **Introdução**

O treinamento de redes neurais com transferência de aprendizagem é uma técnica poderosa que pode melhorar significativamente o desempenho de uma rede neural em uma nova tarefa. Isso pode ser útil quando você tem uma grande quantidade de dados para uma tarefa, mas não tem muitos dados para a outra tarefa.



#### **Como funciona a transferência de aprendizagem?**

Para treinar uma rede neural com transferência de aprendizagem, você primeiro precisa treinar uma rede neural na primeira tarefa. Em seguida, você pode usar os pesos da rede neural treinada como ponto de partida para treinar uma nova rede neural na segunda tarefa.

Os pesos de uma rede neural são os valores que determinam como a rede processa os dados. Ao usar os pesos de uma rede neural treinada como ponto de partida, você pode evitar que a nova rede neural tenha que aprender tudo do zero. Isso pode economizar muito tempo e esforço.



#### **Quando usar a transferência de aprendizagem?**



A transferência de aprendizagem pode ser uma boa opção se você tiver as seguintes condições:

- Você tem uma grande quantidade de dados para uma tarefa, mas não tem muitos dados para outra tarefa.

- As duas tarefas são semelhantes.

  

#### **Como implementar a transferência de aprendizagem?**

Existem várias maneiras de implementar a transferência de aprendizagem. Uma maneira comum é usar uma biblioteca de aprendizado de máquina como TensorFlow ou PyTorch. Essas bibliotecas fornecem funções que facilitam a transferência de pesos de uma rede neural para outra.



#### **Exemplo**

Aqui está um exemplo de como treinar uma rede neural com transferência de aprendizagem usando TensorFlow:

python

```python
import tensorflow as tf

# Carregar a rede neural pré-treinada
pre_trained_model = tf.keras.applications.VGG16(include_top=False, input_shape=(224, 224, 3))

# Congelar as camadas da rede neural pré-treinada
for layer in pre_trained_model.layers:
    layer.trainable = False

# Adicionar novas camadas à rede neural
new_model = tf.keras.Sequential([
    pre_trained_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilar o novo modelo
new_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinar o novo modelo
new_model.fit(x_train, y_train, epochs=10)
```



Neste exemplo, estamos usando uma rede neural VGG16 pré-treinada no conjunto de dados ImageNet. Congelamos as camadas da rede neural pré-treinada para que não sejam atualizadas durante o treinamento. Em seguida, adicionamos novas camadas à rede neural para que possa ser treinada na nova tarefa.



## **Conclusão**



A transferência de aprendizagem é uma técnica poderosa que pode melhorar significativamente o desempenho de uma rede neural em uma nova tarefa. É uma técnica relativamente fácil de implementar e pode ser usada para uma ampla gama de tarefas.



**Códigos adicionais**

Aqui estão alguns códigos adicionais que você pode achar úteis:



- **Carregar um modelo pré-treinado:**

python

```python
pre_trained_model = tf.keras.applications.load_model('path/to/pre_trained_model.h5')
```



- **Congelar as camadas de um modelo:**

python

```python
for layer in model.layers:
    layer.trainable = False
```



- **Adicionar novas camadas a um modelo:**

python

```python
new_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```



- **Compilar um modelo:**

python

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```



- **Treinar um modelo:**

python

```python
model.fit(x_train, y_train, epochs=10)
```





# Treinamento de Rede Neural Artificial





![ezgif com-gif-maker](https://user-images.githubusercontent.com/67600860/136825544-05c9b3d4-d5ce-4e81-a4c9-0b30acb13628.gif)



<p align="justify">
💻 Como trabalho proposto na disciplina Redes Neurais Artificiais da graduação em Engenharia Elétrica, este é um código básico que implementa, passo a passo, o processo de treinamento de uma Perceptron de Múltiplas Camadas por meio do algoritmo backpropagtion. Para isso, foi utilizado um banco de dados conhecido para problemas de classificação, o dataset <a href="https://archive.ics.uci.edu/ml/datasets/iris">Iris</a> 🌼

</p>



## OBS: 

O código tem por objetivo exemplificar a etapa de treino 🏋🏾‍♀ e atualização dos pesos com código próprio, as outras etapas importantes como validação cruzada e teste não estão presentes neste código.





# Modelo de Neurônio

<p align="justify">
O neurônio é a unidade básica morfofuncional do tecido nervoso e, da
mesma maneira, é a unidade básica de processamento da rede neural. Estes
modelos computacionais inspirados em neurônios biológicos possuem
algumas características semelhantes aos naturais para poder desfrutar da
capacidade de aprendizado, são elas: sinais de entrada, pesos sinápticos,
função de soma, função de ativação e sinal de saída.
Para realizar esse processo em uma linguagem de programação, é
aconselhável utilizar operações matriciais, pois assim ocorre uma otimização
na etapa de compilação. Teremos então, um vetor de entrada e um vetor de
pesos (ou matriz, caso haja mais de um neurônio na camada).
</p>


![neuronio](https://user-images.githubusercontent.com/67600860/136829925-bfd180e6-f59b-4901-85d3-a3c2a8d2a98b.png)





# Forma de Aprendizado



<p align="justify">
Os pesos sinápticos são peças fundamentais para o armazenamento de
informação do neurônio artificial, é através deles que o modelo possui a
capacidade de aprender. Utiliza-se o algoritmo de Backpropagation,
que possibilita o ajuste dos pesos em camadas intermediárias, fazendo a
chamada “retropropagação do erro”, ou seja, utiliza o valor do erro da
camada de saída no ajuste dos pesos das outras camadas ocultas.
 </p>




# Algoritmo Backpropagation



<p align="justify">
A inicialização dos pesos sinápticos ocorre de maneira randômica, a
escolha dos pesos iniciais pode prevenir o problema do mínimo local e ainda
a saturação prematura da rede. O fluxo de informação acontece em duas
etapas, primeiramente na etapa "forward", a rede é apresentada aos padrões
de treinamento e tem seus pesos excitados a produzir uma saída, na etapa
“backward”, calcula-se o erro da rede para aquele padrão e este erro, então, é
"retro propagado" para as outras camadas ocultas da rede, que não tem
acesso direto ao erro da camada de saída. É nesta etapa que os pesos são atualizados e a rede adquire conhecimento, o processo se repete até que o
critério de parada seja satisfeito (o critério adotado foi o de número de
épocas de treinamento).
</p>


**A imagem abaixo faz um resumo geral das etapas do código**



<p align="center">
  <img src="https://user-images.githubusercontent.com/67600860/136827442-aff9e61d-ec4c-437f-98d6-1b8e20688ff6.png" />
</p>




# Gráficos



*Os gráficos apresentados a seguir tem por finalidade melhorar a compreensão dos processos executados e a importância de cada um. Para gerar estes gráficos, utilizou-se 5 neurônios na camada escondida, 300 épocas e taxa de aprendizado de 0.01*



* Imagem 1 - Evolução do erro por número de épocas

  

<p align="center">
  <img src="https://user-images.githubusercontent.com/67600860/136830197-fa51fda2-5f17-40d4-9f6f-7149d4c4f9ec.jpg" />
</p>
<p align="justify">
Esta imagem mostra a magnitude do erro quadrático médio para cada número de épocas, onde o erro se inicia em aproximadamente 0.04 e finaliza em aproximadamente 0.015 na época de número 300
</p>



* Imagem 2 - Saída desejada e saída de treinamento

<p align="center">
  <img src="https://user-images.githubusercontent.com/67600860/136830662-dc29ecb8-bbc2-4a29-956f-a7a4e7a25b9a.jpg" />
</p>

<p align="justify">
Apresenta os valores de saída do treinamento (círulo azul) sobrepostos aos valores de saída desejado (asterisco verde) para as classes 1 (Iris Setosa), 2 (Iris
Versicolour) e 3 (Iris Virginica). Note que esta imagem não representa a classificação final, já que os valores estão plotados de maneira contínua. Para performar a classificação, poderíamos por exemplo instituir limites de valores para as classes, sendo a classe 1 correspondendo aos valores de saída entre 0.5 e 1.5, a classe 2 aos valores de 1.5 a 2.5 e a classe 3 aos valores entre 2.5 e 3.5
 </p>



* Imagem 3 - Saídas e épocas

  

<p align="center">
  <img src="https://user-images.githubusercontent.com/67600860/136830937-b0bf0b76-0ec7-4e6d-b08b-50e3bf7ea43b.jpg" />
</p>

<p align="justify">
A finalizade desta imagem é proporcionar o "feeling" sobre como as classificações vão se ajustando e ficando melhor à medida que o número de épocas aumenta. Na imagem podemos observar que as classificações começam muito dispersas e se concentram à medida que os pesos se ajustam
 </p>




## **Características Gerais das Redes Neurais**


Uma rede neural artificial é composta por várias unidades de processamento, cujo funcionamento é bastante simples. Essas unidades, geralmente são conectadas por canais de comunicação que estão associados a determinado peso. As unidades fazem operações apenas sobre seus dados locais, que são entradas recebidas pelas suas conexões. O comportamento inteligente de uma Rede Neural Artificial vem das interações entre as unidades de processamento da rede.



A operação de uma unidade de processamento, proposta por McCullock e Pitts em 1943, pode ser resumida da seguinte maneira:



- Sinais são apresentados à entrada;

  

- Cada sinal é multiplicado por um número, ou peso, que indica a sua influência na saída da unidade;

  

- É feita a soma ponderada dos sinais que produz um nível de atividade;

  

- Se este nível de atividade exceder um certo limite (threshold) a unidade produz uma determinada resposta de saída.

  

![img](https://sites.icmc.usp.br/andre/research/neural/image/mccul.gif)



#### *Esquema de unidade McCullock - Pitts.*



Suponha que tenhamos p sinais de entrada X1, X2, ..., Xp e pesos w1, w2, ..., wp e limitador t; com sinais assumindo valores booleanos (0 ou 1) e pesos valores reais.



#### Neste modelo, o nível de atividade a é dado por:

a = w1X1 + w2X2 + ... + wpXp

A saída y é dada po

- y = 1, se a >= t ou

- y = 0, se a < t.

  

A maioria dos modelos de redes neurais possui alguma regra de treinamento, onde os pesos de suas conexões são ajustados de acordo com os padrões apresentados. Em outras palavras, elas aprendem através de exemplos.

Arquiteturas neurais são tipicamente organizadas em camadas, com unidades que podem estar conectadas às unidades da camada posterior.

![img](https://sites.icmc.usp.br/andre/research/neural/image/camadas_an.gif)

#### *Organização em camadas.

*

Usualmente as camadas são classificadas em três grupos:

- **Camada de Entrada**: onde os padrões são apresentados à rede;

  

- **Camadas Intermediárias ou Escondidas**: onde é feita a maior parte do processamento, através das conexões ponderadas; podem ser consideradas como extratoras de características;

  

- **Camada de Saída**: onde o resultado final é concluído e apresentado.

  

Uma rede neural é especificada, principalmente pela sua topologia, pelas características dos nós e pelas regras de treinamento. A seguir, serão analisados os processos de aprendizado.



------

## **Processos de Aprendizado**


A propriedade mais importante das redes neurais é a habilidade de aprender de seu ambiente e com isso melhorar seu desempenho. Isso é feito através de um processo iterativo de ajustes aplicado a seus pesos, o treinamento. O aprendizado ocorre quando a rede neural atinge uma solução generalizada para uma classe de problemas.

Denomina-se algoritmo de aprendizado a um conjunto de regras bem definidas para a solução de um problema de aprendizado. Existem muitos tipos de algoritmos de aprendizado específicos para determinados modelos de redes neurais, estes algoritmos diferem entre si principalmente pelo modo como os pesos são modificados.

Outro fator importante é a maneira pela qual uma rede neural se relaciona com o ambiente. Nesse contexto existem os seguintes paradigmas de aprendizado:



- **Aprendizado Supervisionado**, quando é utilizado um agente externo que indica à rede a resposta desejada para o padrão de entrada;

  

- **Aprendizado Não Supervisionado** (auto-organização), quando não existe uma agente externo indicando a resposta desejada para os padrões de entrada;

  

- **Reforço**, quando um crítico externo avalia a resposta fornecida pela rede.

Denomina-se ciclo uma apresentação de todos os N pares (entrada e saída) do conjunto de treinamento no processo de aprendizado. A correção dos pesos num ciclo pode ser executado de dois modos:

> **1) Modo Padrão**: A correção dos pesos acontece a cada apresentação à rede de um exemplo do conjunto de treinamento. Cada correção de pesos baseia-se somente no erro do exemplo apresentado naquela iteração. Assim, em cada ciclo ocorrem N correções.



- **2) Modo Batch**: Apenas uma correção é feita por ciclo. Todos os exemplos do conjunto de treinamento são apresentados à rede, seu erro médio é calculado e a partir deste erro fazem-se as correções dos pesos.



------



## **Treinamento Supervisionado**


O treinamento supervisionado do modelo de rede Perceptron, consiste em ajustar os pesos e os thresholds de suas unidades para que a classificação desejada seja obtida. Para a adaptação do threshold juntamente com os pesos podemos considerá-lo como sendo o peso associado a uma conexão, cuja entrada é sempre igual à -1 e adaptar o peso relativo a essa entrada.

Quando um padrão é inicialmente apresentado à rede, ela produz uma saída. Após medir a distância entre a resposta atual e a desejada, são realizados os ajustes apropriados nos pesos das conexões de modo a reduzir esta distância.Este procedimento é conhecido como Regra Delta.





![img](https://sites.icmc.usp.br/andre/research/neural/image/r_delta.jpg)



#### *Regra Delta*



Deste modo, temos o seguinte esquema de treinamento.

Iniciar todas as conexões com pesos aleatórios;

Repita até que o erro E seja satisfatoriamente pequeno (E = e)

Para cada par de treinamento (X,d), faça:

Calcular a resposta obtida O;

Se o erro não for satisfatoriamente pequeno E > e, então:

Atualizar pesos: Wnovo := W anterior + neta E X



### Onde:

- O par de treinamento (X, d) corresponde ao padrão de entrada e a sua respectiva resposta desejada;
- O erro E é definido como: Resposta Desejada - Resposta Obtida (d - O);
- A taxa de aprendizado neta é uma constante positiva, que corresponde à velocidade do aprendizado.

![img](https://sites.icmc.usp.br/andre/research/neural/image/treina.jpg)

*Esquema de treinamento do Perceptron.*

As respostas geradas pelas unidades são calculadas através de uma função de ativação. Existem vários tipos de funções de ativação, as mais comuns são: Hard Limiter, Threshold Logic e Sigmoid.

![img](https://sites.icmc.usp.br/andre/research/neural/image/tra_func.jpg)









#### AGRADECIMENTO ESPECIAL:   

https://github.com/Daniell-Dantas   (parceria e autor de parte)
