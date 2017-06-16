draw_number_of_poems_by_year_chart = function(data) {
  from = min(data$Date) # select the first year for the chart
  to = max(data$Date) # select the last year for the chart
  
  years = seq(from, to) # generate sequence of years between the value of from and to (1820-1870) 
  number_of_poems = c()
  
  for (i in years) {
    poems_in = data[which(data$Date == i),] # select all poems in the year i
    number = nrow(poems_in) # count the poems
    
    number_of_poems = c(number_of_poems, number) # append the number of poems to number_of_poems
  }
  
  plot(x = years, y = number_of_poems, type = "line", xlab = "Year", ylab = "Number of textblocks", col = "lightpink", cex.main = 2, main = "Number of textblocks in training data", lwd = 4) # create the line plot
  points(years, number_of_poems, col = "red", pch = 19)
}

data = read.csv("../data/non-index_poemblocks.csv", sep = "\t")
data$Date = strtoi(format(as.Date(data$Date,'%d/%m/%Y'),'%Y'))
draw_number_of_poems_by_year_chart(data)
