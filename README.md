# Engenharia de Machine Learning
Projeto da disciplina - Infnet


Desenhe um diagrama que demonstra todas as etapas necessárias em um projeto de inteligência artificial desde a aquisição de dados, passando pela criação dos modelos, indo até a operação do modelo

<div align="center">
    <img src="https://github.com/kreuso/engenharia_machine_learning/blob/main/diagrama_projeto_ia.jpg" width="800px"</img> 
</div>
<p><p>
Como as ferramentas Streamlit, MLFlow, PyCaret e Scikit-Learn auxiliam na construção dos pipelines descritos anteriormente?</p>

  O pycaret, como uma ferramenta de AutoML, auxilia no treinamento de modelos utilizando vários "flavours" sendo Scikit-Learn um deles. Ambos, em conjunto também facilitam o pré-processamento com split de treino e teste, declaração de variáveis categóricas, normalização, etc.
  O MLFlow fica na parte de criação e rastreamento de experimentos salvando parâmetros, artefatos para posterior comparações e monitoramento. Empacotamento de códigos de maneira reprodutível facilitando compartilhamento e deploy. Registro, versionamento e armazenamento de modelos e features (model store, feature store), separação dos modelos em staging e production, etc. Também tem compatibilidade com ambientes cloud como Amazon SageMaker e Azure ML.
  O Streamlit já vem como um "front end" auxiliando mais no monitoramento do desempenho do modelo e teste de variáveis. Auxilia a contar a estória por trás da análise de uma maneira mais compreensível.

Foram criados X artefatos na análise, sendo eles:
	<p>AUC: Plota a curva ROC, com objetivo de traçar a taxa de verdadeiro positivo em função da taxa de falso positivo.
	<p>Precision-Recall Curve: Mostrar o trade off entre precision e recall para diferentes thresholds.
  <p>Confusion Matrix: Mostrar números absolutos de verdadeiros e falsos positivos bem como verdadeiros e falsos negativos.
  <p>Feature Importance: Mostrar quais das variáveis mais impactam a decisão do modelo.
  <p>Learning Curve: Mostrar se resultados de treino e teste convergem para um mesmo lugar, prevenindo overfitting ou underfitting.
		
		
