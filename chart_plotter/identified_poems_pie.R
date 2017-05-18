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

data = read.csv("data/found_poems.csv")

#draw_pie_charts(data, years = min(data$Year):max(data$Year))
draw_overall_pie_chart(data)