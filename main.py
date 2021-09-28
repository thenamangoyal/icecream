import os
import time
import math
import numpy as np
import remi.gui as gui
from remi import start, App
import copy
import json

class IceCreamGame(App):
    def __init__(self, *args):
        res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res')
        self.ice_cream_container = IceCreamContainer()
        self.l = self.ice_cream_container.get_length()
        self.w = self.ice_cream_container.get_width()
        self.h = self.ice_cream_container.get_height()

        self.flavors = self.ice_cream_container.get_flavors()
        
        self.players = []
        self.player_names = []
        self.served = []
        self.turns_received = np.zeros(0, dtype=np.int)
        self.players_score = []
        self.next_player = -1
        
        self.max_allowed_per_turn = 24
        self.total_turn_per_player = -1

        self.__add_player(Player(np.random.permutation(self.flavors).tolist()), "Group 1")
        self.__add_player(Player(np.random.permutation(self.flavors).tolist()), "Group 2")
        self.__add_player(Player(np.random.permutation(self.flavors).tolist()), "Group 3")
        self.__add_player(Player(np.random.permutation(self.flavors).tolist()), "Group 4")

        self.next_player = self.__assign_next_player()
        
        super(IceCreamGame, self).__init__(*args, static_file_path={'res':res_path})
    
    # def idle(self):
    #     self.update_table()
    #     self.update_score_table()

    def __scoop(self, i, j):
        # self.label.set_text("Scooping {}".format((i,j)))
        scooped_items = self.ice_cream_container.scoop(i,j, dry_run=False)
        # self.label.set_text("Scooped {}".format(scooped_items))
        return scooped_items
    
    def __add_player(self, p, name):
        if name not in self.player_names:
            self.players.append(p)
            self.player_names.append(name)
            self.served.append({k:0 for k in self.flavors})
            self.turns_received = np.zeros(len(self.players), dtype=np.int)
            self.players_score.append(0)
            self.total_turn_per_player = math.floor(120/ len(self.players))
        else:
            print("Failed to insert player as another player with name {} exists.".format(name))
    
    def __assign_next_player(self):
        # find first min turns recevied player
       return np.argmin(self.turns_received)

    def __game_end(self):
        print("Game finished")
        self.label.set_text("Game ended, as each player played: {} turns".format(self.total_turn_per_player))

    def __play_all(self):
        self.label.set_text("Playing all turns")
        while np.amin(self.turns_received) < self.total_turn_per_player:
            self.__play(do_update=False)
        self.update_score_table()
        self.update_table()
        self.__game_end()
    
    def __play(self, do_update=True):
        if np.amin(self.turns_received) < self.total_turn_per_player:
            if np.amin(self.turns_received) < self.turns_received[self.next_player]:
                print("Can't pass to the player {}, as there are other players who have received less number of helps".format(self.next_player))
                self.next_player = self.__assign_next_player()

            print("Passed to player {}".format(self.next_player))
            self.label.set_text("Current turn Player {}".format(self.next_player))
            new_next_player = self.__turn_p(self.next_player, do_update)
            if new_next_player is not None:
                self.next_player = new_next_player
            else:
                self.next_player = self.__assign_next_player()
            self.label.set_text("Next turn Player {}".format(self.next_player))
        else:
            self.__game_end()


    def __turn_p(self, player_idx, do_update=True):
        assert player_idx < len(self.players) and player_idx >= 0, "Player idx out of bounds"
        self.turns_received[player_idx] += 1

        player = self.players[player_idx]
        served_this_turn = []
        next_player = None
        
        while len(served_this_turn) < self.max_allowed_per_turn:
            top_layer = self.ice_cream_container.get_top_layer()
            curr_level = self.ice_cream_container.get_curr_level()
            action_values_dict = player.serve(top_layer, curr_level, player_idx, self.get_flavors, self.get_player_count, self.get_served, self.get_turns_received)
            action = action_values_dict["action"]
            values = action_values_dict["values"]
            print("Received action: {} from player {}".format(action_values_dict, player_idx))
            
            if action == "scoop":
                i, j = values
                if len(self.ice_cream_container.scoop(i,j, dry_run=True)) + len(served_this_turn) <= self.max_allowed_per_turn:
                    scooped_items = self.__scoop(i,j)
                    for flavor in scooped_items:
                        self.served[player_idx][flavor] += 1
                        self.players_score[player_idx] += len(self.flavors) - player.get_flavor_preference(flavor) + 1

                    served_this_turn.extend(scooped_items)
                else:
                    break
            elif action == "pass":
                if values < 0 or values >= len(self.players):
                    print("Next player {} is out of bounds".format(values))
                elif values == player_idx:
                    print("Can't ask to pass to yourself")
                else:
                    next_player = values
                break
        if do_update:
            self.update_score_table()
            self.update_table()
        
        return next_player

    def get_flavors(self):
        return self.flavors
    
    def get_player_count(self):
        return len(self.players)
    
    def get_player_names(self):
        return copy.deepcopy(self.player_names)
    
    def get_served(self):
        return copy.deepcopy(self.served)

    def get_turns_received(self):
        return np.copy(self.turns_received)

    def main(self):
        mainContainer = gui.Container(style={'width': '100%', 'display': 'block', 'overflow': 'auto', 'text-align': 'center'})
        mainContainer.style['justify-content'] = 'center'
        mainContainer.style['align-items'] = 'center'

        bt_hbox = gui.HBox(width="30%", style={'text-align': 'center', 'margin': 'auto'})
        play_bt = gui.Button("Play")
        play_all_bt = gui.Button("Play All")
        bt_hbox.append([play_bt, play_all_bt])

        play_bt.onclick.do(self.play_bt_press)
        play_all_bt.onclick.do(self.play_all_bt_press)
        mainContainer.append(bt_hbox)
        self.label = gui.Label("Ice Cream: Ready to start")
        mainContainer.append(self.label)

        self.score_table = gui.TableWidget(2, len(self.players), style={'margin':'5px auto'})
        self.update_score_table()
        for player_idx, _ in enumerate(self.players_score):
            self.score_table.item_at(0, player_idx).set_style("padding:0 10px")
            self.score_table.item_at(1, player_idx).set_style("padding:0 10px")
        mainContainer.append(self.score_table)

        paths = ["gray.png", "yellow.png", "green.png", "red.png", "blue.png", "cream.png", "pink.png", "orange.png", "brown.png", "cyan.png", "almond.png", "strawberry.png"]
        self.flavor_to_path = {k:v for k,v in zip(self.flavors, paths)}
        self.flavor_to_path[-1] = "cross.png"

        self.table =  gui.VBox(width='100%', height='100%', style={'margin':'0px auto'})
        self.tablerows = [gui.HBox(width='100%', height='100%', style={'margin':'0px auto'}) for _ in range(self.l)]
        
        for j in range(self.w):
            for i in range(self.l):
                self.tablerows[j].append(gui.Image("/res:{}".format(self.flavor_to_path[-1])), key=i)
                self.tablerows[j].style['justify-content'] = 'center'
                self.tablerows[j].style['align-items'] = 'center'
            self.table.append(self.tablerows[j], key=j)
            self.table.style['justify-content'] = 'center'
            self.table

        mainContainer.append(self.table)

        self.update_table()

        return mainContainer
    
    def update_score_table(self):
        for player_idx, score in enumerate(self.players_score):
            self.score_table.item_at(0, player_idx).set_text("{}".format(self.player_names[player_idx]))
            self.score_table.item_at(1, player_idx).set_text("{}, {}".format(score, self.turns_received[player_idx]))

    def update_table(self):
        top_layer = self.ice_cream_container.get_top_layer()
        curr_level = self.ice_cream_container.get_curr_level()
        for j in range(self.w):
            for i in range(self.l):
                main_div = gui.Widget(style="background-image: url(/res:{})".format(self.flavor_to_path[top_layer[i,j]]))
                main_div.set_size("46px", "46px")
                border = "{}px black".format(self.h - curr_level[i,j] -1)
                main_div.set_style("-webkit-box-shadow:inset 0px 0px 0px {}; -moz-box-shadow:inset 0px 0px 0px {}; box-shadow:inset 0px 0px 0px {}".format(border, border, border))
                # main_div.set_style("text-align: right")
                # label = gui.Label("{}".format(curr_level+1))
                # main_div.add_child(0, label)
                self.set_cell(i, j, main_div)
    
    def set_cell(self, i, j, widget):
        self.tablerows[j].append(widget, key=i)

    def fill_table(self, values):
        for ri in range(0, self.table.row_count):
            for ci in range(0, self.table.column_count):
                self.table.item_at(ri, ci).set_text("{}".format(values[ri, ci]))

    def play_bt_press(self, widget):
        self.__play()
    
    def play_all_bt_press(self, widget):
        self.__play_all()


class IceCreamContainer:
    def __init__(self) -> None:
        self.flavors = list(range(1,13))
        self.l = 24 # cols
        self.w = 15 # rows
        self.h = 8 # height
        self.container = np.empty((self.l, self.w, self.h), dtype=np.int)
        self.curr_level = np.empty((self.l, self.w), dtype=np.int)

        self.possible_types = np.array([2,3,4,5,6,8,9,10,12], dtype=np.int)
        self.ice_cream_type = self.possible_types[np.random.randint(0, self.possible_types.size)]

        print("Using ice cream type {}".format(self.ice_cream_type))
        with open(os.path.join("types", "{}.json".format(self.ice_cream_type)), "r") as jf:
            flavor_assigned = np.array(json.load(jf), dtype=np.int)
        
        self.container = np.copy(flavor_assigned)

        for j in range(self.w):
            for i in range(self.l):
                self.curr_level[i,j] = self.h-1

    def get_flavors(self):
        return self.flavors

    def get_length(self):
        return self.l

    def get_width(self):
        return self.w
    
    def get_height(self):
        return self.h

    def get_top_layer(self):
        top_layer = np.empty((self.l, self.w), dtype=np.int)
        for j in range(self.w):
            for i in range(self.l):
                if self.curr_level[i,j] >= 0:
                    k = self.curr_level[i,j]
                    top_layer[i,j] = self.container[i,j,k]
                else:
                    top_layer[i,j] = -1
        return top_layer
    
    def get_curr_level(self):
        return np.copy(self.curr_level)

    def scoop(self, i, j, dry_run=False):
        assert i >= 0 and i < self.l-1, "i value {} for scooping out of bounds".format(i)
        assert j >= 0 and j < self.w-1, "j value {} for scooping out of bounds".format(j)
        
        scooped_items = []
        for iter_j in range(j, j+2):
            for iter_i in range(i, i+2):
                iter_k = self.curr_level[iter_i,iter_j]
                if iter_k >= 0:
                    scooped_items.append(self.container[iter_i, iter_j, iter_k])
                    if not dry_run:
                        self.curr_level[iter_i,iter_j] += -1
        
        return scooped_items




class Player:
    def __init__(self, flavor_preference) -> None:
        self.flavor_preference = flavor_preference
        self.state = None
        self.initialize()

    def initialize(self):
        pass

    def get_flavor_preference(self, flavor):
        return self.flavor_preference.index(flavor) + 1
    
    def serve(self, top_layer, curr_level, player_idx, get_flavors, get_player_count, get_served, get_turns_received):
        x = np.random.randint(0,4)
        if x < 3:
            # print(top_layer.shape)
            i = np.random.randint(0, top_layer.shape[0]-1)
            j = np.random.randint(0, top_layer.shape[1]-1)
            action = "scoop"
            values = (i,j)
        else:
            other_player_list = list(range(0,get_player_count()))
            other_player_list.remove(player_idx)
            next_player = other_player_list[np.random.randint(0, len(other_player_list))]
            action = "pass"
            values = next_player
        return {"action": action,  "values" : values}

if __name__ == '__main__':
    
    # start(IceCreamGame, port=80, start_browser=False)
    start(IceCreamGame, address='0.0.0.0', port=8080, start_browser=False)
