scores = [3,4,6,1,2,8,4,6,2,1,3,0]
# for i in range(len(scores)):
#     i = 0
#     while(scores[i]<scores[i+1]):
#             tmp = scores[i]
#             scores[i] = scores[i+1]
#             scores[i+1] = tmp
#             i += 1



n = len(scores)
for i in range(n):
    for j in range(n-i-1):
        if scores[j] < scores[j+1]:
            scores[j], scores[j+1] = scores[j+1], scores[j]

print(scores)