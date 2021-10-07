import os
import time
import math
import numpy as np
import remi.gui as gui
from remi import start, App
import copy
import json
import logging
import argparse
from random_player import Player as RandomPlayer

root_dir = os.path.dirname(os.path.abspath(__file__))


class IceCreamGame(App):
    def __init__(self, args):
        self.args = args
        self.use_gui = not(args.no_gui)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(os.path.join(root_dir, 'debug.log'), mode="w")
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)
        fh.setFormatter(logging.Formatter('%(message)s'))

        self.rng = np.random.default_rng(2021)
        self.ice_cream_container = IceCreamContainer(self.rng, self.logger)
        self.l = self.ice_cream_container.get_length()
        self.w = self.ice_cream_container.get_width()
        self.h = self.ice_cream_container.get_height()

        self.flavors = self.ice_cream_container.get_flavors()

        self.players = []
        self.player_names = []
        self.player_preferences = []
        self.served = []
        self.turns_received = np.zeros(0, dtype=np.int)
        self.player_scores = []
        self.next_player = -1

        self.max_allowed_per_turn = 24
        self.total_turn_per_player = -1

        self.__add_player(RandomPlayer, "Group 1")
        self.__add_player(RandomPlayer, "Group 2")
        self.__add_player(RandomPlayer, "Group 3")
        self.__add_player(RandomPlayer, "Group 4")
        self.__add_player(RandomPlayer, "Group 5")
        self.__add_player(RandomPlayer, "Group 6")

        self.next_player = self.__assign_next_player()
        self.processing_turn = False
        self.served_this_turn = None

        if self.use_gui:
            start(IceCreamApp, address=args.address, port=args.port, start_browser=not(args.no_browser), userdata=(self, None))
        else:
            self.logger.debug("No GUI flag specified")
            self.play_all()

    def __log(self, message):
        self.logger.debug(message)
        if self.use_gui:
            self.ice_cream_app.label_set_text(message)

    def __add_player(self, player_class, player_name):
        if player_name not in self.player_names:
            player_preference = self.rng.permutation(self.flavors).tolist()
            player = player_class(player_preference, self.rng, self.logger)
            self.players.append(player)
            self.player_preferences.append(player_preference)
            self.player_names.append(player_name)
            self.served.append({k: 0 for k in self.flavors})
            self.turns_received = np.zeros(len(self.players), dtype=np.int)
            self.player_scores.append(0)
            self.total_turn_per_player = math.floor(120 / len(self.players))
        else:
            self.logger.debug("Failed to insert player as another player with name {} exists.".format(player_name))

    def __assign_next_player(self):
        # randomly select among valid players
        least_helpings = np.amin(self.turns_received)
        valid_players = np.argwhere(self.turns_received == least_helpings)
        return valid_players[self.rng.integers(0, valid_players.size)][0]

    def __game_end(self):
        self.__log("Game ended as each player played {} turns".format(self.total_turn_per_player))
        for player_idx, score in enumerate(self.player_scores):
            self.logger.debug("{} turns: {}".format(self.player_names[player_idx], self.turns_received[player_idx]))
        for player_idx, score in enumerate(self.player_scores):
            self.logger.debug("{} individual score: {}".format(self.player_names[player_idx], score))
        group_score = np.mean(self.player_scores)
        self.logger.debug("Average group score: {}".format(group_score))
        for player_idx, score in enumerate(self.player_scores):
            self.logger.debug("{} final score: {}".format(self.player_names[player_idx], np.mean([score, group_score])))

    def __turn_end(self, new_next_player=None):
        self.processing_turn = False
        self.served_this_turn = None

        if new_next_player is not None:
            if new_next_player < 0 or new_next_player >= len(self.players):
                self.__log("Can't pass to player idx {}, as out of bounds".format(new_next_player))
                self.next_player = self.__assign_next_player()
                self.logger.debug("Assigned new player {}".format(self.player_names[self.next_player]))
            elif np.amin(self.turns_received) < self.turns_received[new_next_player]:
                self.__log("Can't pass to the {}, as other player(s) with less helpings exist".format(self.player_names[new_next_player]))
                self.next_player = self.__assign_next_player()
                self.logger.debug("Assigned new player {}".format(self.player_names[self.next_player]))
            else:
                self.__log("{} passed to {}".format(self.player_names[self.next_player], self.player_names[new_next_player]))
                self.next_player = new_next_player
        else:
            self.__log("No next player specified by {}".format(self.player_names[self.next_player]))
            self.next_player = self.__assign_next_player()
            self.logger.debug("Assigned new player {}".format(self.player_names[self.next_player]))
        self.logger.debug("Next turn {}".format(self.player_names[self.next_player]))
        if self.use_gui:
            self.ice_cream_app.label_set_text("{}, Next turn {}".format(self.ice_cream_app.label_get_text(), self.player_names[self.next_player]))

    def set_app(self, ice_cream_app):
        if self.use_gui:
            self.ice_cream_app = ice_cream_app
        else:
            self.logger.debug("No GUI flag specified, skipping setting app")

    def play_all(self):
        self.__log("Playing all turns")
        while np.amin(self.turns_received) < self.total_turn_per_player:
            self.play(run_stepwise=False, do_update=False)
        if self.use_gui:
            self.ice_cream_app.update_score_table()
            self.ice_cream_app.update_table()
        self.__game_end()

    def play(self, run_stepwise=False, do_update=True):
        if not self.processing_turn:
            if np.amin(self.turns_received) < self.total_turn_per_player:
                if np.amin(self.turns_received) < self.turns_received[self.next_player]:
                    self.logger.debug("Can't pass to the {}, as other player(s) with less helpings exist".format(self.player_names[self.next_player]))
                    self.next_player = self.__assign_next_player()
                    self.logger.debug("Assigned new player {}".format(self.player_names[self.next_player]))

                self.__log("Current turn {}".format(self.player_names[self.next_player]))

                self.processing_turn = True
                self.served_this_turn = []
                self.turns_received[self.next_player] += 1

            else:
                self.__game_end()
                return

        if run_stepwise:
            pass_next, new_next_player = self.__step_p(self.next_player, do_update)
            if pass_next:
                self.__turn_end(new_next_player)

        else:
            pass_next = False
            while not pass_next:
                pass_next, new_next_player = self.__step_p(self.next_player, do_update=False)
            if do_update and self.use_gui:
                self.ice_cream_app.update_score_table()
                self.ice_cream_app.update_table()
            self.__turn_end(new_next_player)

    def __check_action(self, action_values_dict):
        is_valid = False
        if isinstance(action_values_dict, dict):
            if set(action_values_dict.keys()) == set(["action", "values"]):
                if action_values_dict["action"] in ["scoop", "pass"]:
                    if action_values_dict["action"] == "scoop":
                        if len(action_values_dict["values"]) == 2:
                            if np.issubdtype(type(action_values_dict["values"][0]), np.int) and np.issubdtype(type(action_values_dict["values"][1]), np.int):
                                is_valid = True
                    elif action_values_dict["action"] == "pass":
                        if np.issubdtype(type(action_values_dict["values"]), np.int):
                            is_valid = True
        return is_valid

    def __step_p(self, player_idx, do_update=True):
        pass_next = False
        next_player = None
        if len(self.served_this_turn) <= self.max_allowed_per_turn:
            player = self.players[player_idx]
            top_layer = self.ice_cream_container.get_top_layer()
            curr_level = self.ice_cream_container.get_curr_level()
            action_values_dict = player.serve(top_layer, curr_level, player_idx, self.get_flavors, self.get_player_count, self.get_served, self.get_turns_received)
            is_valid_action = self.__check_action(action_values_dict)
            if is_valid_action:
                action = action_values_dict["action"]
                values = action_values_dict["values"]
                self.logger.debug("Received action: {} from {}".format(action_values_dict, self.player_names[player_idx]))
                if self.use_gui:
                    self.ice_cream_app.label_set_text("{}, {}".format(self.ice_cream_app.label_get_text(), action_values_dict))

                if action == "scoop":
                    i, j = values
                    if not(i >= 0 and i < self.l-1 and j >= 0 and j < self.w-1):
                        self.logger.debug("Given out of bounds scoop position {}".format((i, j)))
                        pass_next = True
                    elif len(self.ice_cream_container.scoop(i, j, dry_run=True)) <= 0:
                        self.logger.debug("Given empty scooping position, passing to next player")
                        pass_next = True
                    elif len(self.ice_cream_container.scoop(i, j, dry_run=True)) + len(self.served_this_turn) <= self.max_allowed_per_turn:
                        scooped_items = self.ice_cream_container.scoop(i, j, dry_run=False)
                        for flavor in scooped_items:
                            self.served[player_idx][flavor] += 1
                            self.player_scores[player_idx] += len(self.flavors) - self.__get_flavor_preference(player_idx, flavor) + 1

                        self.served_this_turn.extend(scooped_items)
                    else:
                        self.logger.debug("Scooping limit exceeded, passing to next player")
                        pass_next = True
                elif action == "pass":
                    if values < 0 or values >= len(self.players):
                        self.logger.debug("Next player idx {} is out of bounds".format(values))
                    elif values == player_idx:
                        self.logger.debug("Can't ask to pass to yourself")
                    else:
                        next_player = values
                    pass_next = True
            else:
                self.logger.debug("Given invalid action_value_dict.")
                pass_next = True

            if do_update and self.use_gui:
                self.ice_cream_app.update_score_table()
                self.ice_cream_app.update_table()
        else:
            pass_next = True
        return pass_next, next_player

    def __get_flavor_preference(self, player_idx, flavor):
        return self.player_preferences[player_idx].index(flavor) + 1

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


class IceCreamApp(App):
    def __init__(self, *args):
        res_path = os.path.join(root_dir, 'res')
        super(IceCreamApp, self).__init__(*args, static_file_path={'res': res_path})

    def main(self, *userdata):
        self.ice_cream_game, _ = userdata
        self.ice_cream_game.set_app(self)

        mainContainer = gui.Container(style={'width': '100%', 'display': 'block', 'overflow': 'auto', 'text-align': 'center'})
        mainContainer.style['justify-content'] = 'center'
        mainContainer.style['align-items'] = 'center'

        bt_hbox = gui.HBox(width="30%", style={'text-align': 'center', 'margin': 'auto'})
        play_step_bt = gui.Button("Play Step")
        play_turn_bt = gui.Button("Play Turn")
        play_all_bt = gui.Button("Play All")
        bt_hbox.append([play_step_bt, play_turn_bt, play_all_bt])

        play_step_bt.onclick.do(self.play_step_bt_press)
        play_turn_bt.onclick.do(self.play_turn_bt_press)
        play_all_bt.onclick.do(self.play_all_bt_press)
        mainContainer.append(bt_hbox)
        self.label = gui.Label("Ice Cream: Ready to start")
        mainContainer.append(self.label)

        self.score_table = gui.TableWidget(2, len(self.ice_cream_game.players)+1, style={'margin': '5px auto'})
        self.update_score_table()

        for player_idx, _ in enumerate(self.ice_cream_game.player_scores):
            self.score_table.item_at(0, player_idx).set_style("padding:0 10px")
            self.score_table.item_at(1, player_idx).set_style("padding:0 10px")
        mainContainer.append(self.score_table)

        paths = ["gray.png", "yellow.png", "green.png", "red.png", "blue.png", "cream.png", "pink.png", "orange.png", "brown.png", "cyan.png", "almond.png", "strawberry.png"]
        self.flavor_to_path = {k: v for k, v in zip(self.ice_cream_game.flavors, paths)}
        self.flavor_to_path[-1] = "cross.png"

        self.table = gui.VBox(width='100%', height='100%', style={'margin': '0px auto'})
        self.tablerows = [gui.HBox(width='100%', height='100%', style={'margin': '0px auto'}) for _ in range(self.ice_cream_game.l)]

        for j in range(self.ice_cream_game.w):
            for i in range(self.ice_cream_game.l):
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
        for player_idx, score in enumerate(self.ice_cream_game.player_scores):
            self.score_table.item_at(0, player_idx).set_text("{}".format(self.ice_cream_game.player_names[player_idx]))
            self.score_table.item_at(1, player_idx).set_text("{}, {}".format(score, self.ice_cream_game.turns_received[player_idx]))
        self.score_table.item_at(0, len(self.ice_cream_game.players)).set_text("{}".format("Average"))
        self.score_table.item_at(1, len(self.ice_cream_game.players)).set_text("{:.2f}".format(np.mean(self.ice_cream_game.player_scores)))

    def update_table(self):
        top_layer = self.ice_cream_game.ice_cream_container.get_top_layer()
        curr_level = self.ice_cream_game.ice_cream_container.get_curr_level()
        for j in range(self.ice_cream_game.w):
            for i in range(self.ice_cream_game.l):
                main_div = gui.Widget()
                main_div.set_style({"background": "url(/res:{})".format(self.flavor_to_path[top_layer[i, j]])})
                main_div.set_size("46px", "46px")
                border = "4px rgba(0,0,0,{})".format((self.ice_cream_game.h - curr_level[i, j]-1) / (self.ice_cream_game.h))
                main_div.set_style("-webkit-box-shadow:inset 0px 0px 0px {}; -moz-box-shadow:inset 0px 0px 0px {}; box-shadow:inset 0px 0px 0px {}".format(border, border, border))

                label = gui.Label()
                label.attributes["class"] = "hoverable"
                label.set_text("{}".format(curr_level[i, j]+1))
                main_div.add_child("label", label)

                self.set_cell(i, j, main_div)

    def set_cell(self, i, j, widget):
        self.tablerows[j].append(widget, key=i)

    def fill_table(self, values):
        for ri in range(0, self.table.row_count):
            for ci in range(0, self.table.column_count):
                self.table.item_at(ri, ci).set_text("{}".format(values[ri, ci]))

    def play_step_bt_press(self, widget):
        self.ice_cream_game.play(run_stepwise=True)

    def play_turn_bt_press(self, widget):
        self.ice_cream_game.play(run_stepwise=False)

    def play_all_bt_press(self, widget):
        self.ice_cream_game.play_all()

    def label_set_text(self, text):
        self.label.set_text(text)

    def label_get_text(self):
        return self.label.get_text()


class IceCreamContainer:
    def __init__(self, rng, logger, num_flavors=12) -> None:
        self.rng = rng
        self.logger = logger
        if num_flavors not in [2, 3, 4, 5, 6, 8, 9, 10, 12]:
            self.logger.debug("Num flavors {} is not in allowed values, using 12 flavors".format(num_flavors))
            num_flavors = 12
        self.flavors = list(range(1, num_flavors+1))
        self.l = 24  # cols
        self.w = 15  # rows
        self.h = 8  # height
        self.container = np.empty((self.l, self.w, self.h), dtype=np.int)
        self.curr_level = np.empty((self.l, self.w), dtype=np.int)

        self.container = self.generate_type()

        for j in range(self.w):
            for i in range(self.l):
                self.curr_level[i, j] = self.h-1

    def generate_type(self):
        empty_container = np.empty((self.l, self.w, self.h), dtype=np.int)

        assert empty_container.size % len(self.flavors) == 0, "Number of flavors can't divide container size"
        flavor_repeated = np.repeat(np.array(self.flavors, dtype=np.int), empty_container.size / len(self.flavors))

        flavor_assigned = self.rng.permutation(flavor_repeated)

        return np.reshape(flavor_assigned, empty_container.shape)

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
                if self.curr_level[i, j] >= 0:
                    k = self.curr_level[i, j]
                    top_layer[i, j] = self.container[i, j, k]
                else:
                    top_layer[i, j] = -1
        return top_layer

    def get_curr_level(self):
        return np.copy(self.curr_level)

    def scoop(self, i, j, dry_run=False):
        scooped_items = []

        if i >= 0 and i < self.l-1 and j >= 0 and j < self.w-1:
            heights = []
            for iter_j in range(j, j+2):
                for iter_i in range(i, i+2):
                    iter_k = self.curr_level[iter_i, iter_j]
                    heights.append(iter_k)
            max_depth = np.amax(np.array(heights, dtype=np.int))
            for iter_j in range(j, j+2):
                for iter_i in range(i, i+2):
                    iter_k = self.curr_level[iter_i, iter_j]
                    if iter_k >= 0 and max_depth == iter_k:
                        scooped_items.append(self.container[iter_i, iter_j, iter_k])
                        if not dry_run:
                            self.curr_level[iter_i, iter_j] += -1

        return scooped_items


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", "-p", type=int, default=8080, help="Port to start")
    parser.add_argument("--address", "-a", type=str, default="127.0.0.1", help="Address")
    parser.add_argument("--no_browser", "-nb", action="store_true", help="Disable browser launching in GUI mode")
    parser.add_argument("--no_gui", "-ng", action="store_true", help="Disable GUI")
    args = parser.parse_args()
    ice_cream_game = IceCreamGame(args)
