library(STRINGdb)
library(igraph)
library(parallel)


getPPI <- function(expr, species, genFlag='symbol'){
  
  genList <- rownames(expr)
  genList <- toupper(genList)
  
  if(species=='mouse'){
    
    string_db <- STRINGdb$new(version = "11.5", species = 10090, score_threshold = 200)
    
  }else{
    
    string_db <- STRINGdb$new(version = "11.5", species = 9606, score_threshold = 200)
    
  }
  
  if(genFlag!='symbol'){
    message('Gene names should be converted first!')
  }
  
  string_id <- data.frame(genes=genList)
  genes_mapped <- string_db$map(string_id, "genes" ,removeUnmappedRows = TRUE)
  graph <- string_db$get_subnetwork(genes_mapped$STRING_id)
  
  ppi_network <- as_adjacency_matrix(graph)
  ppi_network <- as.matrix(ppi_network)
  
  gn <- genes_mapped[match(rownames(ppi_network),genes_mapped[,2]),1]
  rownames(ppi_network) <- gn
  colnames(ppi_network) <- gn
  
  intgn <- intersect(gn, rownames(expr))
  
  ppi <- matrix(0, nrow = nrow(expr), ncol = nrow(expr), dimnames = list(rownames(expr), rownames(expr)))
  
  ppi[intgn, intgn] <- ppi_network[intgn, intgn]
  
  return(ppi)
  
}


compJIppi <- function(dat, threshfilt = 0.02, numcores = 5){
  
  G2 <- (dat!=0)
  L2.jc <- matrix(0, nrow = nrow(dat), ncol = nrow(dat))
  
  getJc <- function(i) {
    if(i %% 100 == 0){
      print(i)
    }
    
    g1 <- (dat[i, ] != 0)
    G1 <- matrix(g1, nrow = nrow(dat), ncol = nrow(dat), byrow = TRUE)
    # L2.jc[i, ] <<- rowSums(G1 & G2) / rowSums(G1 | G2)
    
    res <- list()
    res$geneindex <- i
    res$weights <- rowSums(G1 & G2) / rowSums(G1 | G2)
    res$weights[is.nan(res$weights)] <- 0
    
    return(res)
  }
  
  jc <- mclapply(1:nrow(dat), getJc, mc.cores = numcores)
  
  geneord <- unlist(lapply(jc, `[[`, 1))
  
  if(all(geneord == 1:nrow(dat))){
    ppi <- lapply(jc, `[[`, 2)
    ppi <- do.call(rbind, ppi)
    rownames(ppi) <- rownames(dat)
    colnames(ppi) <- rownames(dat)
  }
  
  allnames <- lapply(jc, function(x){
    names(x[[2]])
  })
  
  prenames <- allnames[[1]]
  allnamescheck <- unlist(lapply(2:length(allnames), function(i){
    curnames <- allnames[[i]]
    identical(prenames, curnames)
  }))
  
  mat <- matrix(0, nrow = length(geneord), ncol = length(geneord))
  
  for(i in 1:nrow(mat)){
    mat[geneord[i],] <- jc[[i]][[2]]
  }
  
  if(identical(mat, t(mat))){
    print('good sym check! ')
    if(all(allnamescheck)){
      print('good name check! ')
      colnames(mat) <- prenames
      rownames(mat) <- prenames
    }
  }
  
  if(identical(mat, ppi)){
    print('finish double check')
  }
  
  mat[mat <= threshfilt] <- 0
  
  diag(mat) <- 0
  
  D <- colSums(mat)
  mat <- diag(D) - mat
  
  mat
}
