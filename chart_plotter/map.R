library(maps)       # Provides functions that let us plot the maps
library(mapdata)    # Contains the hi-resolution points that mark out the countries.

issues_in_year = function(data, year) {
  issn = unique(data$ISSN[which(data$Year == year)])
  
  if(length(issn) == 0) {
    return(NA)
  }
  
  count = c()
  for(i in 1:length(issn)) {
    count = c(count, sum(data$ISSN[which(data$Year == year)] == issn[i]))
  }
  
  return(data.frame(issn, count)[order(count), ])
}

data = read.csv("data/found_poems.csv")
places = read.csv("data/publication_places.csv")

for(i in min(data$Year):max(data$Year)) {
  issues = issues_in_year(data, i)
  
  if(is.na(issues))
    next
  
  png(paste(i, ".png", sep = ""))
  map('worldHires', c('Finland'))
  title(main = paste("Year", i, sep = ' '))
  
  for(j in 1:nrow(issues)) {
    publication_places = places[which(as.character(places$issn) == as.character(issues$issn[j])),]
    
    ratio = issues$count[j] / max(issues$count)
    publication_place = publication_places[1,]
    
    points(publication_place$long, publication_place$lat, col = "red", cex = 1 + ratio, pch = 19)
    text(publication_place$long, publication_place$lat, labels = issues$count[j], cex= 0.2 + ratio, pos = 4)
  }
}

dev.off()