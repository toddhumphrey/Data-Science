#Please note that one of the key things that makes this useful is that we have the code for the book from Joel Grus availabe in this workspace
#No need to spend hours trying to code every line of code that is available.  That is not productive.
#really, should copy and paste any lists that are available, and when it comes to cementing concepts, try to do it yourself.  no need to memorize
#adding comments to each graph that is created in order to avoid confusion when trying to create the charts


from matplotlib import pyplot as plt 

#years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
#gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]

# create a line chart, years on x-axis, gdp on y-axis
#plt.plot(years, gdp, color='green', marker='o', linestyle='solid')

# add a title
#plt.title("Nominal GDP")

# add a label to the y-axis
#plt.ylabel("Billions of $")
#plt.show()



##Bar Charts are good to show how quantity varies among discrete set of items

#movies = ["Annie Hall", "Ben-Hur", "Casablanca", "Gandhi", "West Side Story"]
#num_oscars = [5, 11, 3, 8, 10]

#Plotting bars with left x-coordinates [0, 1, 2, 3, 4], heights [num_oscars]
#plt.bar(range(len(movies)), num_oscars)
#plt.title("My Fav Movies") #adding title
#plt.ylabel("# of Academy Awards") #adding axis label
#plt.xticks(range(len(movies)), movies)

#plt.show()



from collections import Counter
grades = [83, 95, 91, 87, 70, 0, 85, 82, 100, 67, 73, 77, 0]

# Bucket grades by decile, but put 100 in with the 90s
#histogram = Counter(min(grade // 10 * 10, 90) for grade in grades)

#plt.bar([x + 5 for x in histogram.keys()],  # Shift bars right by 5
        #histogram.values(),                 # Give each bar its correct height
        #10,                                 # Give each bar a width of 8
        #edgecolor=(0, 0, 0))                # Black edges for each bar

#plt.axis([-5, 105, 0, 5])                  # x-axis from -5 to 105, first bar is not snug to the x axis
                                           # y-axis from 0 to 5

#plt.xticks([10 * i for i in range(11)])    # x-axis labels at 0, 10, ..., 100
#plt.xlabel("Decile")
#plt.ylabel("# of Students")
#plt.title("Distribution of Exam 1 Grades")
#plt.show()



##Be mindful when using the .axis( ) for the bar charts since it can be misleading when the y axis does not start at zero

#mentions = [500, 505]
#years = [2017, 2018]

#plt.bar(years, mentions, 0.8)
#plt.xticks(years)
#plt.ylabel("# of times I heard someone say 'data science'")

# if you don't do this, matplotlib will label the x-axis 0, 1
# and then add a +2.013e3 off in the corner (bad matplotlib!)
#plt.ticklabel_format(useOffset=False)

# misleading y-axis only shows the part above 500
#plt.axis([2016.5, 2018.5, 499, 506])
#plt.title("Look at the 'Huge' Increase!")
#plt.show()
##This graph mislabels a huge increase in something that is absolutely not a huge increase

## If you show a graph with more sensible axes, looks less impressive
#plt.axis([2016.5, 2018.5, 0, 550])
#plt.title("Not So Huge Anymore!")
#plt.show()



#Line charts are apporpriate for showing trends over time

variance     = [1, 2, 4, 8, 16, 32, 64, 128, 256]
bias_squared = [256, 128, 64, 32, 16, 8, 4, 2, 1]
total_error  = [x + y for x, y in zip(variance, bias_squared)]
xs = [i for i, _ in enumerate(variance)]

# We can make multiple calls to plt.plot
# to show multiple series on the same chart
plt.plot(xs, variance,     'g-',  label='variance')    # green solid line
plt.plot(xs, bias_squared, 'r-.', label='bias^2')      # red dot-dashed line
plt.plot(xs, total_error,  'b:',  label='total error') # blue dotted line

# Because we've assigned labels to each series,
# we can get a legend for free (loc=9 means "top center") [Don't forget "loc" means location]
plt.legend(loc=9)
plt.xlabel("model complexity")
plt.xticks([])
plt.title("The Bias-Variance Tradeoff")
#plt.show()

plt.gca().clear()



#Scatterplots are good for showing the relationship between two paired sets of data
#Something to be wary of: Scatterplot can be misleading if you let MatPlotLib choose the scale

friends = [ 70,  65,  72,  63,  71,  64,  60,  64,  67]
minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]
labels =  ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

plt.scatter(friends, minutes)

# label each point
for label, friend_count, minute_count in zip(labels, friends, minutes):
    plt.annotate(label,
        xy=(friend_count, minute_count), # Put the label with its point
        xytext=(5, -5),                  # but slightly offset
        textcoords='offset points')

plt.title("Daily Minutes vs. Number of Friends")
plt.xlabel("# of friends")
plt.ylabel("daily minutes spent on the site")
#plt.show()

plt.gca().clear()

test_1_grades = [ 99, 90, 85, 97, 80]
test_2_grades = [100, 85, 60, 90, 70]

plt.scatter(test_1_grades, test_2_grades)
plt.title("Axes Aren't Comparable")
plt.xlabel("test 1 grade")
plt.ylabel("test 2 grade")
#plt.axis("Equal") #When included, this allows the variotion display to be more accurate
plt.show()

#For further studying on visualizations, see:
    #MatPlotLib Gallery
    #Seaborn