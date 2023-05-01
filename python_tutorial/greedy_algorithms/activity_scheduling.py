# custom class to set up start-time, end-time and meeting position
class Activity:
    def __init__(self, start, end, pos):
        self.start = start
        self.end = end
        self.pos = pos


# function to find maximum number of activities
def schedule_activity(activity_list, num_activities):
    # sort activities based on their finish times
    activity_list.sort(key=lambda a:  a.end)

    # list to account activity positions of max activity scheduler
    max_activity = []

    max_activity.append(activity_list[0].pos)

    time_limit = activity_list[0].end
    for i in range(1, num_activities):
        if activity_list[i].start >= time_limit:
            max_activity.append(activity_list[i].pos)
            time_limit = activity_list[i].end

    # Print maximum meeting positions
    for i in range(len(max_activity)):
        print(max_activity[i]+1, end=" ")


# driver code
s = [1, 3, 0, 5, 8, 5]
f = [2, 4, 6, 7, 9, 9]

num_activities = len(s)
print("Number of activities:", num_activities)

activity_list = []
for i in range(num_activities):
    activity_list.append(Activity(s[i], f[i], i))

schedule_activity(activity_list, num_activities)

