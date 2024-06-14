% BACKPROPAGATION
clear; close all; clc;

%% Carregar os dados de conjuntos de treinamento, teste e valida��o do banco de dados

%PADR�ES DE ENTRADA
load iris_treino.txt;
I1=iris_treino'; % Carregar os padr�es de treino do banco de dados Iris com os atributos e a respectiva classe

I=I1(1:4, :); % Defini��o do vetor de entrada para ser normalizado

%% Requisi��es ao usu�rio e par�metros

%Pedir o n�mero de neuronios da camada oculta
Neuronios_oculta=input('Ner�nios na Camada Oculta: ');%
%Especificar o n�mero de neuronios da camada de sa�da, no nosso caso � 1
Neuronios_saida = 1;
%Pedir o n�mero de �pocas
epocas = input('Digite o n�mero de �pocas: ');
%Pedir a taxa de aprendizagem
taxa_aprendizagem = input('Digite a taxa de aprendizagem: ');
E = exp(-10); %Definir precis�o (limite de erro)


%% Normaliza��o
normalizado = ((I - min(min(I))*(1))/(max(max(I)) - min(min(I)))); %considerando Lmax = 1 e Lmin = 0.

%% Organizando as matrizes

%Matriz de treinamento e sua respectiva sa�da
Entrada_treinamento = normalizado(:,:);
Saida_desejada_treinamento = I1(5, :);
[numero_linhas_treinamento, numero_colunas_treinamento] = size(Entrada_treinamento); %Define o n�mero de linhas e colunas da matriz de entrada no treinamento. Precisamos desses valores para fazer al�gica do Foward.


%% Inicializar os valores dos pesos da camada 1 e 2
w1 = rand(Neuronios_oculta,numero_linhas_treinamento+1); %PESOS DA CAMADA OCULTA
w2 = rand(Neuronios_saida,Neuronios_oculta+1); %PESOS DA CAMADA DE SA�DA

%% Defini��o das Fun��es de ativa��o e derivadas usando "function_handle"
Sigmoide = @(z)1./(1 + exp(-z)); %FUN��O DE ATIVA��O SIGMOIDE NA CAMADA OCULTA
deriv_sigmoide = @(z)(exp(z)/(1+exp(z)).^2); %DERIVADA DA SIGMOIDE
linear = @(z) z; % FUN��O DE ATIVA��O LINEAR NA CAMADA DE SA�DA
deriv_linear = @(z) 1; %DERIVADA DA FUN��O LINEAR


%% ETAPA DE TREINAMENTO
ww1=w1; %Definimos os novos pesos ww1 e ww2 para poder guardar o valor dos pesos originais e mostrar os pesos finais
ww2=w2;
Saida_treinamento = zeros(1, numero_colunas_treinamento);
EQM_train = zeros(1, epocas);
Saida_treinamento_total = zeros(epocas, 90);

for a = 1:epocas %Crit�rio de parada. Define que o la�o come�a em 1 e segue at� o valor de "epocas". A vari�vel parada guarda a itera��o atual
Erro_quadratico=0;
    for i = 1:numero_colunas_treinamento %Estamos percorrendo todas as 90 colunas, ou seja, os 90 padr�es para o treino
    %~~~~~~~~~~~~~~~~~~ IN�CIO FOWARD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    %~~~~~~~ CAMADA DE ENTRADA ~~~~~~~
    x=[-1;Entrada_treinamento(:,i)]; 
    Ij1 = ww1*x; 
    Yj1 = Sigmoide(Ij1); %Fun��o de ativa��o da camada 1 (oculta): Foi escolhida a fun��o Sigm�ide. Esta � a sa�da dos neur�nios da camada oculta

    %~~~~~~~ CAMADA DE SA�DA ~~~~~~~
    y=[-1; Yj1]; %Aqui � a sa�da dos neur�nios da camada oculta + bias. 
    Ij2 = ww2*y;  % Entradas para o neur�nio da camada de sa�da
    Yj2 = linear(Ij2); %Fun��o de ativa��o da camada de sa�da: Foi escolhida a fun��o Linear. Esta � a sa�da do neur�nio da camada de sa�da, e portanto, da rede


    %~~~~~~~~~~~~~~~~~~ IN�CIO BACKWARD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    erro = Saida_desejada_treinamento(:,i) - Yj2; %Sinal de erro da camada de sa�da

    Erro_quadratico = Erro_quadratico + 0.5*sum(erro.^2); % Considerando todos os neur�mnios da camada de sa�da, a soma do erro quadr�tico no instante N, ou seja, para um par entrada sa�da. No nosso caso, a soma ter� o valor de somente 1 erro.
    derivada_saida_rede = deriv_linear(Ij2); %Derivada da fun��o linear na camada de sa�da
    gradiente_local_saida = erro.*derivada_saida_rede; %Gradiente local da camada de saida.

    % Ajuste dos pesos na camada de sa�da
    ww2 = ww2 + taxa_aprendizagem*gradiente_local_saida*y';

    derivada_saida_camada_oculta = deriv_sigmoide(Ij1); %Derivada da fun��o sigm�ide na camada oculta
    gradiente_local_oculta = derivada_saida_camada_oculta*(ww2(:,2:end)'*gradiente_local_saida) ; % GRADIENTE LOCAL DA CAMADA OCULTA

    % Ajuste dos pesos na camda oculta
    ww1 = ww1 + taxa_aprendizagem*gradiente_local_oculta*x';

    Saida_treinamento(i) = Yj2;
    

    end %90
    
Saida_treinamento_total(a,:) = Saida_treinamento;
EQM_train(a) = sum(Erro_quadratico)/numero_colunas_treinamento; %Agora considerando todos os N momentos, ou seja, os N pares de entrada e sa�da que eu tenho no banco da dados. Esta � a fun��o custo, o nosso erro quadr�tico m�dio, a que queremos minimzar durante o treinamento.

end %300

%% PLOTAGEM DOS GR�FICOS
figure(1);
plot(EQM_train,'LineWidth',2) % PLOTA ERRO QUADR�TICO M�DIO DE TREINAMENTO
legend('ERRO DE TREINO')
title('EVOLU��O DO ERRO')
ylabel('ERRO QUADR�TICO M�DIO')
xlabel('�POCAS')
grid on

figure(2);
plot(Saida_desejada_treinamento, 'g*')
hold on;
plot(Saida_treinamento, 'bo')
legend('Sa�da desejada', 'Sa�da do treinamento')
title('SA�DA DESEJADA X SA�DA TREINAMENTO')
ylabel('CLASSES')
xlabel('DADOS')
grid on

figure(3);
plot(Saida_treinamento_total)
title('SA�DA X N�MERO DE �POCAS')
ylabel('SA�DAS')
xlabel('�POCAS')
grid on

