# Engenharia de Machine Learning
Projeto da disciplina - Infnet


Desenhe um diagrama que demonstra todas as etapas necessárias em um projeto de inteligência artificial desde a aquisição de dados, passando pela criação dos modelos, indo até a operação do modelo

<div align="center">
    <img src="https://github.com/kreuso/engenharia_machine_learning/blob/main/diagrama_projeto_ia.jpg" width="800px"</img> 
</div>
<p><p>
Como as ferramentas Streamlit, MLFlow, PyCaret e Scikit-Learn auxiliam na construção dos pipelines descritos anteriormente?</p>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;O pycaret, como uma ferramenta de AutoML, auxilia no treinamento de modelos utilizando vários "flavours" sendo Scikit-Learn um deles. Ambos, em conjunto também facilitam o pré-processamento com split de treino e teste, declaração de variáveis categóricas, normalização, etc.
  O MLFlow fica na parte de criação e rastreamento de experimentos salvando parâmetros, artefatos para posterior comparações e monitoramento. Empacotamento de códigos de maneira reprodutível facilitando compartilhamento e deploy. Registro, versionamento e armazenamento de modelos e features (model store, feature store), separação dos modelos em staging e production, etc. Também tem compatibilidade com ambientes cloud como Amazon SageMaker e Azure ML.
  O Streamlit já vem como um "front end" auxiliando mais no monitoramento do desempenho do modelo e teste de variáveis. Auxilia a contar a estória por trás da análise de uma maneira mais compreensível.

Foram criados 5 artefatos na análise, sendo eles:

 - <b>AUC</b>: Plota a curva ROC, com objetivo de traçar a taxa de verdadeiro positivo em função da taxa de falso positivo.
 - <b>Precision-Recall Curve</b>: Mostrar o trade off entre precision e recall para diferentes thresholds.
 - <b>Confusion Matrix</b>: Mostrar números absolutos de verdadeiros e falsos positivos bem como verdadeiros e falsos negativos.
 - <b>Feature Importance</b>: Mostrar quais das variáveis mais impactam a decisão do modelo.
 - <b>Learning Curve</b>: Mostrar se resultados de treino e teste convergem para um mesmo lugar, prevenindo overfitting ou underfitting.
		
3 modelos foram inseridos no pycaret para escolha de um destaque pela acurácia: Regressão Logística, Decision Tree e Support Vector Machines. Estes foram escolhidos por serem os modelos já implementados em outra disciplina. Neste caso, a regressão logística teve melhor desempenho e partiu para <i>hypertunning</i>.

O modelo em operação não mostrou bom resultado, visto que foi treinado para arremessos de 2 pontos e passou a operar para arremessos de 3 pontos. Certamente a distância teve um papel importante que pode ter sido negligenciado no dataset de treino. Abaixo um Classification Report do modelo em operação:

<div align="center">
    <img src="https://github.com/kreuso/engenharia_machine_learning/blob/main/arremesso_3_pts_classification_report.png" width="400px"</img> 
</div>

Caso haja disponibilidade da variável resposta, o modelo pode ser monitorado e retreinado por estratégia reativa, isto é, sempre que o desempenho cair abaixo de um limite pré-estabelecido, um alarme é acionado indicando o retreino para posterior homologação e produção.
Em caso de não disponibilidade da variável resposta, o modelo pode ser monitorado e retreinado por estratégia reativa, isto é, um intervalo de tempo é pré-estabelecido para realização do retreino.
