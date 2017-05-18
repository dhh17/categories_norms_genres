draw_number_of_poems_by_year_chart = function(data) {
  from = min(data$Year) # select the first year for the chart
  to = max(data$Year) # select the last year for the chart
  
  years = seq(from, to) # generate sequence of years between the value of from and to (1820-1870) 
  number_of_poems = c()
  
  for (i in years) {
    poems_in = data[which(data$Year == i),] # select all poems in the year i
    number = nrow(poems_in) # count the poems
    
    number_of_poems = c(number_of_poems, number) # append the number of poems to number_of_poems
  }
  
  plot(x = years, y = number_of_poems, type = "line", xlab = "Year", ylab = "Number of poems", col = "red") # create the line plot
  
  article_index = read.csv("data/poem_count_from_article_index.csv", sep = "\t")
  article_index = article_index[which(article_index$Year <= max(years)),] # reducing the data
  lines(article_index$Year, article_index$Poems, col = "blue")
  
  differences = abs(article_index$Poems - number_of_poems) # calculate the differences in poems between the amount predicted by the algorithm and the count valeus from the article index
  cat("Year with the biggest difference:", years[which(differences == max(differences))]) 
}

data = read.csv("data/found_poems.csv")
draw_number_of_poems_by_year_chart(data)