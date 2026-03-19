from cmath import sqrt


def hillClimbing(startP, points, times):
    pointsCopy = points.copy()
    startPCopy = startP
    pointsCopy.remove(startPCopy)
    route = [startP]
    height = 0
    for i in range(times):
        #pick neighbor
        print(f'hill climbing iteration {i}, current point: {startPCopy}, remaining points: {pointsCopy}')
        bestNeighbor = neighbor(pointsCopy, startPCopy)
        print(bestNeighbor)

        height += sqrt(bestNeighbor[1])
        route.append(bestNeighbor[0])
        pointsCopy.remove(bestNeighbor[0])
        if pointsCopy == []:
            print('No more points to explore, stopping hill climbing.')
            break
        startPCopy = bestNeighbor[0]
    print('route:' + ' => '.join(str(point) for point in route))
    print(f'height: {height:.2f}')

def neighbor(points, startP):
    pointsCopy = points.copy()
    print(f'Finding neighbor for {startP} in {pointsCopy}')
    bestNeighbor = []
    currentDist = 0
    for point in pointsCopy:
        x = point[0] - startP[0]
        y = point[1] - startP[1]
        currentDist = x**2 + y**2
        if bestNeighbor == []:
            print(f'bestNeighbor is empty, set to {point}')
            bestNeighbor = [point, currentDist]
        elif currentDist < bestNeighbor[1]:
            print(f'Found better neighbor: {point}')
            bestNeighbor[0] = point
            bestNeighbor[1] = currentDist
    return bestNeighbor

if __name__ == "__main__":
    points = [[1,1], [1,4], [3,7], [2,2], [5,3], [4,5], [6,2], [7,4], [8,1], [9,6]]
    hillClimbing(points[4], points, 10)