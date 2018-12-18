import matplotlib.pyplot as plt 
 

filename = "spikes.dat"
with open(filename) as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content]

x = []

for ind in range(0,len(content)):
	x.append(ind)

plt.plot(x, content, label = "line") 
  
# # naming the x axis 
plt.xlabel('x - axis') 
# # naming the y axis 
plt.ylabel('y - axis') 
# # giving a title to my graph 
plt.title('Trace') 
  
# # show a legend on the plot 
plt.legend() 
  
# # function to show the plot 
plt.show() 