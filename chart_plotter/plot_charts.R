issues_in_year = function(data, year) {
  newspaper = unique(data$Newspaper.name[which(data$Year == year)])
  
  if(length(newspaper) == 0) {
    return(NA)
  }
  
  count = c()
  for(i in 1:length(newspaper)) {
    count = c(count, sum(data$Newspaper.name[which(data$Year == year)] == newspaper[i]))
  }
  
  return(data.frame(newspaper, count)[order(count), ])
}

issues_per_newspaper = function(data) {
  newspaper = unique(data$Newspaper.name)
  
  if(length(newspaper) == 0) {
    return(NA)
  }
  
  count = c()
  for(i in 1:length(newspaper)) {
    count = c(count, sum(data$Newspaper.name == newspaper[i]))
  }
  
  return(data.frame(newspaper, count)[order(count), ])
}

draw_pie_charts = function(data, years) {
  for(i in 1:length(years)) {
    draw_pie_chart(data = data, year = years[i])
  }
}

draw_pie_chart = function(data, year) {
  values = issues_in_year(data, year)
  
  if(is.na(values) == FALSE) {
    pie(values$count, labels = values$newspaper, main = year) 
  }
}

draw_overall_pie_chart = function(data) {
  values = issues_per_newspaper(data)
  
  if(is.na(values) == FALSE) {
    colors = colours()[1:nrow(values)]
    percentages = paste(round(values$count / sum(values$count) * 100, digits = 2), "% ", "(", values$count, ")", sep="")
    pie(values$count, labels = percentages, main = "Identified poems (whole corpus)", init.angle = 90, col = colors) 
    legend("topleft", legend = rev(values$newspaper), cex = 0.75, fill = rev(colors), bty = "n")
  }
}

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
#draw_pie_charts(data, years = min(data$Year):max(data$Year))
draw_overall_pie_chart(data)
