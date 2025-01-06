class Game:
  def __init__(self,name: str,players: list[str]):
    self.name = name
    self.players = players
    self.player_count = len(players)
  
  def add_player(self, new_player: str ):
    self.players.append(new_player)
    self.player_count = len(self.players)

volleyball = Game("Volleyball",["t","a","b"])

active = volleyball

volleyball.add_player(["m"])
print(active.name + ": " + str(active.player_count) + " players")

def print_linewise(text):
  line = ''
  for letter in text:
    if letter == '\n':
      print(line)
      line = ''
    else:
      line += letter