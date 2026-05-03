# Buscador-de-roupas

Projeto para busca de imagens similares na base [Deepfashion2](https://github.com/switchablenorms/deepfashion2)

## Documentação

O código create_df_experiment.py usa o diretório deepfashion2/train com os dados de treinamento para criar arquivos csv para treino, teste e validação
com a referência da imagem, categoria e coordenadas da bounding box para os experimentos 1, 2 e 3.

com o experiments criado o código deepfashion2_to_yolo.py transforma esses dados para o formato usado pelo YOLO, com isso o modelo pode ser treinado com train.py

O código create_embeddings.py cria os arquivos com embeddings de todos os objetos da base, um para cada categoria, que será usado como galeria para busca de similaridade

### Para usar

criar e ativar venv no linux

```bash
python3 -m venv .venv && source .venv/bin/activate
```

ou no windows

```bash
python -m venv .venv
.venv\Scripts\Activate
```

e instalar as dependências

```bash
pip install -r requirements.txt
```

para usar a interface de usuário, entre na pasta app e ative a API com:

```bash
fastapi dev
```

depois entre na pasta frontend e copie o conteúdo de .env.example em .env, para ativar o nextJS:

```bash
npm run dev
```

e acesse http://localhost:3000/
