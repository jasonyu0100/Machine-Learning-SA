class Game:
    ROWS = 8
    COLS = 8
    EMPTY = "."
    def __init__(self):
        self.grid = self.constructGrid()
        self.turnNum = 0

    def __str__(self):
        string = ""
        string += 'O   ' + ' '.join([str(num) for num in range(len(self.grid))]) + '\n'
        string += '   ' + (len(self.grid)*2) * '_' + '\n'
        for rowNum,row in enumerate(self.grid):
            string += str(rowNum) + ' |'
            for col in row:

                if type(col) == BaseCounter:
                    string += ' '+(col.color)
                else:
                    string += ' '+col
            string += '\n'
        return string

    def constructGrid(self):
        grid = []
        for r in range(self.ROWS):
            row = []
            for c in range(self.COLS):
                row.append(self.EMPTY)
            grid.append(row)

        grid = self.initPieces(grid)
        return grid

    def initPieces(self,grid):
        currentColor = "W"
        for row in [0,1]:
            offset = row % 2
            for col in range(self.COLS):
                if col % 2 != offset:
                    grid[row][col] = BaseCounter(row,col,currentColor,self)

        currentColor = "B"
        for row in [6,7]:
            offset = row % 2
            for col in range(self.COLS):
                if col % 2 != offset:
                    grid[row][col] = BaseCounter(row,col,currentColor,self)
        return grid

    def getCounter(self):
        while(True):
            r,c = [int(i) for i in input("Enter Counter Coordinate: ").split(',')]
            current = self.grid[r][c]
            if type(current) == BaseCounter:
                return current
            else:
                print("Enter a valid coord")


    def startGame(self):
        print(self)
        while (True):
            if (self.turnNum % 2 == 0):
                print("White's Turn")
            else:
                print("White's Turn")
            counter = self.getCounter()
            print(counter.validMoves)
            command = self.command(counter)
            if command == False:
                continue
            else:
                self.turnNum += 1
                print(self)

    def command(self,counter):
        nR,nC = [int(num) for num in input("Enter a Move Coordinate ").split(',')]
        if (nR,nC) in counter.validMoves:
            counter.move(nR,nC)
            print("Counter moved to Coordinate\nRow: {0}, Col: {1}".format(nR,nC))
            return True
        else:
            print("Coordinate not in counter range")
            return False

class BaseCounter:
    def __init__(self,r,c,color,game):
        self.game = game
        self.r = r
        self.c = c
        self.color = self.setColor(color)
        self.vectors = self.setVectors()

    def __str__(self):
        return self.color

    def setColor(self,color):
        if color != "W" and color != "B":
            raise ValueError("Color is not valid")
        else:
            return color

    def setVectors(self):
        if self.color == "W":
            return [(1,1),(1,-1)]
        elif self.color == "B":
            return [(-1,1),(-1,-1)]
        else:
            raise ValueError("Color is not valid")

    def move(self,nR,nC):
        self.game.grid[self.r][self.c] = Game.EMPTY
        self.r = nR
        self.c = nC
        self.game.grid[self.r][self.c] = self

    @property
    def validMoves(self):
        return self.getRange()

    def getRange(self):
        coordRange = set()
        for changeR,changeC in self.vectors:
            newR = self.r + changeR
            newC = self.c + changeC
            if newR < 0 or newR > len(self.game.grid) or newC < 0 or newC > len(self.game.grid):
                continue
            else:
                coordRange.add((newR,newC))
        return coordRange

class PromotedCounter(BaseCounter):
    def __init__(self,r,c,color,game):
        super.__init__(self,r,c,color,game)
        self.vectors = [(-1,1),(-1,-1),(1,1),(1,-1)]

    #def getRange(self):

def main():
    game = Game()
    game.startGame()


if __name__ == "__main__":
    main()







    

