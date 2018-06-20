

def check_sparsity(matrix):
    # Compute Sparsicity = Percentage of Non-Zero cells
    return print("Sparsicity (матрица разряженности нулей и неНулей): ", ((matrix > 0).sum()/matrix.size)*100, "%")

def check_likelihood(ldaModel, dataVectorized):
# Log Likelyhood: Higher the better
    return print("Log Likelihood (Higher is better): ", ldaModel.score(dataVectorized))

def check_perplexity(ldaModel, dataVectorized):
# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
    return print("Perplexity (Lower is better): ", ldaModel.perplexity(dataVectorized))