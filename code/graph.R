
data = read.csv("/Users/ysz/Downloads/300k.csv")

#generate edge file for pokemon appearance graph
edge = data.frame(source = numeric(), target = numeric(), count = numeric())
for (i in 1:151) {
  sub = data[data$pokemonId == i, 57:207]
  for (j in 1:151) {
    count = sum(sub[, j] == "true")
    if (count > 0) {
      edge = rbind(edge, data.frame(source = i, target = j, count = count))
    }
  }
}
write.csv(edge, file = "/Users/ysz/Documents/bda_hw3/pkm_edge.csv", row.names = FALSE, quote = FALSE)

#generate node file for pokemon appearance graph
names = read.csv("/Users/ysz/Downloads/pokemon.csv")
types = read.csv("/Users/ysz/Downloads/types.csv")
type_links = read.csv("/Users/ysz/Downloads/pokemon_types.csv")
node = data.frame(id = numeric(), name = character(), type = character())
for (i in 1:151) {
  name = toString(names[i, 2])
  type_sub = type_links[type_links$pokemon_id == i, ]
  if (dim(type_sub)[1] == 1) {
    type = toString(types[type_sub[1, 2], 2])
  } else {
    type = paste(toString(types[type_sub[1, 2], 2]), toString(types[type_sub[2, 2], 2]), sep = "/")
  }
  node = rbind(node, data.frame(id = i, name = name, type = type))
}
write.csv(node, file = "/Users/ysz/Documents/bda_hw3/pkm_node.csv", row.names = FALSE, quote = FALSE)

#generate node file for pokemon type graph
type_names = c("normal",
               "fighting",
               "flying",
               "poison",
               "ground",
               "rock",
               "bug",
               "ghost",
               "steel",
               "fire",
               "water",
               "grass",
               "electric",
               "psychic",
               "ice",
               "dragon",
               "fairy")
for (type in type_names) {
  for (type in type_names) {
    type_edge
  }
}
type_edge = data.frame(source = character(), target = character(), count = numeric())
for (type in type_names) {
  for (i in 1:dim(edge)[1]) {
    source_type = toString(node[edge[i, 1], 3])
    target_type = toString(node[edge[i, 2], 3])
    count = edge[i, 3]
    type_edge = rbind(type_edge, data.frame(source = source_type, target = target_type, count = count))
  }
}

#draw plot
scores = read.csv('/Users/ysz/Documents/bda_hw3/knn_scores.csv')
library(ggplot2)
p1 <- ggplot(scores, aes(x = k, y = percent_score, group = p))
p1 + geom_line(aes(colour = p), size=0.5) + scale_colour_gradient(low="red") + ylim(60, 100) + xlab("number of neighbors") + ylab("KNN score") + labs(title = "KNN accuracy")

ggplot(data=scores, aes(x=k,y=score,group=p,colour=p)) + geom_line()
