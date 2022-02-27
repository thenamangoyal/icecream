import os
import time
import signal
import math
import numpy as np
import remi.gui as gui
from remi import start, App
import copy
import json
import traceback
import logging
import argparse
import constants
from players.random_player import Player as Random_Player
from players.g1_player import Player as G1_Player
from players.g2_player import Player as G2_Player
from players.g3_player import Player as G3_Player
from players.g4_player import Player as G4_Player
from players.g5_player import Player as G5_Player
from players.g6_player import Player as G6_Player
from players.g7_player import Player as G7_Player
from players.g8_player import Player as G8_Player
from players.g9_player import Player as G9_Player
from players.g10_player import Player as G10_Player

root_dir = os.path.dirname(os.path.abspath(__file__))
return_vals = ["player_names", "final_scores", "winner_list", "player_scores", "player_preferences", "served", "total_time_sorted", "turns_received", "timeout_count", "error_count"]


class TimeoutException(Exception):   # Custom exception class
    pass


def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException


class MainLoggingFilter(logging.Filter):
    def __init__(self, name: str) -> None:
        super().__init__(name=name)

    def filter(self, record):
        if record.name == self.name:
            return True
        else:
            return False


class PlayerLoggingFilter(logging.Filter):
    def __init__(self, name: str) -> None:
        super().__init__(name=name)

    def filter(self, record):
        if self.name in record.name or record.name == __name__:
            return True
        else:
            return False


class IceCreamGame():
    def __init__(self, player_list, args):
        self.use_gui = not(args.no_gui)
        self.do_logging = not(args.disable_logging)
        if not self.use_gui:
            self.use_timeout = not(args.disable_timeout)
        else:
            self.use_timeout = False

        self.logger = logging.getLogger(__name__)
        # create file handler which logs even debug messages
        if self.do_logging:
            self.logger.setLevel(logging.DEBUG)
            self.log_dir = os.path.abspath(args.log_path)
            os.makedirs(self.log_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(self.log_dir, 'debug.log'), mode="w")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter('%(message)s'))
            fh.addFilter(MainLoggingFilter(__name__))
            self.logger.addHandler(fh)
            result_path = os.path.join(self.log_dir, "results.log")
            rfh = logging.FileHandler(result_path, mode="w")
            rfh.setLevel(logging.INFO)
            rfh.setFormatter(logging.Formatter('%(message)s'))
            rfh.addFilter(MainLoggingFilter(__name__))
            self.logger.addHandler(rfh)
        else:
            if args.log_path:
                self.logger.setLevel(logging.INFO)
                result_path = os.path.abspath(args.log_path)
                self.log_dir = os.path.dirname(result_path)
                os.makedirs(self.log_dir, exist_ok=True)
                rfh = logging.FileHandler(result_path, mode="w")
                rfh.setLevel(logging.INFO)
                rfh.setFormatter(logging.Formatter('%(message)s'))
                rfh.addFilter(MainLoggingFilter(__name__))
                self.logger.addHandler(rfh)
            else:
                self.logger.setLevel(logging.ERROR)
                self.logger.disabled = True

        if args.seed == 0:
            args.seed = None
            self.logger.info("Initialise random number generator with no seed")
        else:
            self.logger.info("Initialise random number generator with {} seed".format(args.seed))
        self.rng = np.random.default_rng(args.seed)
        if args.flavors == 0:
            self.logger.info("Using random number of flavors")
            rand_num_flavors_idx = self.rng.integers(0, len(constants.num_flavor_choices))
            args.flavors = constants.num_flavor_choices[rand_num_flavors_idx]
        self.ice_cream_container = IceCreamContainer(self.rng, self.logger, num_flavors=args.flavors)
        self.l = self.ice_cream_container.get_length()
        self.w = self.ice_cream_container.get_width()
        self.h = self.ice_cream_container.get_height()

        self.flavors = self.ice_cream_container.get_flavors()

        self.players = []
        self.player_names = []
        self.player_preferences = []
        self.served = []
        self.time_taken = []
        self.turns_received = np.zeros(0, dtype=np.int)
        self.timeout_count = np.zeros(0, dtype=np.int)
        self.error_count = np.zeros(0, dtype=np.int)
        self.player_scores = []
        self.next_player = -1

        self.max_allowed_per_turn = constants.max_allowed_per_turn
        self.total_turn_per_player = -1

        self.__add_players(player_list)

        self.next_player = self.__assign_next_player()
        self.processing_turn = False
        self.served_this_turn = None
        self.end_message_printed = False

        self.final_scores = None
        self.winner_list = None
        self.total_time_sorted = None

        if self.use_gui:
            start(IceCreamApp, address=args.address, port=args.port, start_browser=not(args.no_browser), update_interval=0.5, userdata=(self, args.automatic))
        else:
            self.logger.debug("No GUI flag specified")

    def get_state(self):
        return_dict = dict()
        for val in return_vals:
            value = getattr(self, val)
            if isinstance(value, np.ndarray):
                value = value.tolist()
            return_dict[val] = value
        return return_dict

    def __log(self, message, label_num=0):
        self.logger.debug(message)
        if self.use_gui:
            self.ice_cream_app.set_label_text(message, label_num)

    def __add_players(self, player_list):
        player_count = dict()
        for player_name in player_list:
            if player_name not in player_count:
                player_count[player_name] = 0
            player_count[player_name] += 1

        count_used = {k: 0 for k in player_count}
        for player_name in player_list:
            if player_name in constants.possible_players:
                if player_name.lower() == "r":
                    player_class = Random_Player
                    base_player_name = "Random"
                else:
                    player_class = eval("G{}_Player".format(player_name))
                    base_player_name = "Group {}".format(player_name)
                count_used[player_name] += 1
                if player_count[player_name] == 1:
                    self.__add_player(player_class, "{}".format(base_player_name))
                else:
                    self.__add_player(player_class, "{}.{}".format(base_player_name, count_used[player_name]))
            else:
                self.logger.error("Failed to insert player {} since invalid player name provided.".format(player_name))

    def __add_player(self, player_class, player_name):
        if player_name not in self.player_names:
            player_preference = self.rng.permutation(self.flavors).tolist()
            self.logger.info("Adding player {} from class {} with preference {}".format(player_name, player_class.__module__, player_preference))
            player = player_class(player_preference, self.rng, self.__get_player_logger(player_name))
            self.players.append(player)
            self.player_preferences.append(player_preference)
            self.player_names.append(player_name)
            self.served.append({k: 0 for k in self.flavors})
            self.time_taken.append([])  # a list of time taken in each turn (recorded as list of step times) for a player
            self.turns_received = np.zeros(len(self.players), dtype=np.int)
            self.timeout_count = np.zeros(len(self.players), dtype=np.int)
            self.error_count = np.zeros(len(self.players), dtype=np.int)
            self.player_scores.append(0)
            self.total_turn_per_player = math.floor(constants.max_total_turns / len(self.players))
        else:
            self.logger.error("Failed to insert player as another player with name {} exists.".format(player_name))

    def __get_player_logger(self, player_name):
        player_logger = logging.getLogger("{}.{}".format(__name__, player_name))

        if self.do_logging:
            player_logger.setLevel(logging.INFO)
            # add handler to self.logger with filtering
            player_fh = logging.FileHandler(os.path.join(self.log_dir, '{}.log'.format(player_name)), mode="w")
            player_fh.setLevel(logging.DEBUG)
            player_fh.setFormatter(logging.Formatter('%(message)s'))
            player_fh.addFilter(PlayerLoggingFilter(player_name))
            self.logger.addHandler(player_fh)
        else:
            player_logger.setLevel(logging.ERROR)
            player_logger.disabled = True

        return player_logger

    def __assign_next_player(self):
        # randomly select among valid players
        least_helpings = np.amin(self.turns_received)
        valid_players = np.argwhere(self.turns_received == least_helpings)
        return valid_players[self.rng.integers(0, valid_players.size)][0]

    def __game_end(self):
        if not self.end_message_printed and self.is_game_ended():
            self.end_message_printed = True
            self.logger.info("Game ended as each player played {} turns".format(self.
                                                                                total_turn_per_player))
            if self.use_gui:
                self.ice_cream_app.set_label_text("Game ended as each player played {} turns".format(self.total_turn_per_player))
            for player_idx, score in enumerate(self.player_scores):
                self.logger.debug("{} turns: {}".format(self.player_names[player_idx], self.turns_received[player_idx]))
            total_time = np.zeros(len(self.players))
            for player_idx, player_time_taken in enumerate(self.time_taken):
                player_time_taken_flatten = np.concatenate(player_time_taken)
                if player_time_taken_flatten.size == 0:
                    player_time_taken_flatten = np.zeros(1)
                self.logger.info("{} took {} steps, total time {:.3f}s, avg step time {:.3f}s, max step time {:.3f}s".format(self.player_names[player_idx], player_time_taken_flatten.size, np.sum(player_time_taken_flatten), np.mean(player_time_taken_flatten), np.amax(player_time_taken_flatten)))
                total_time[player_idx] = np.sum(player_time_taken_flatten)
            self.logger.info("Total time taken by all players {:.3f}s".format(np.sum(total_time)))
            total_time_sort_idx = np.argsort(total_time)[::-1]
            self.total_time_sorted = [(self.player_names[player_idx], total_time[player_idx]) for player_idx in total_time_sort_idx]
            self.logger.info("Players sorted by total time")
            for (player_name, player_total_time) in self.total_time_sorted:
                self.logger.info("{} took {:.3f}s".format(player_name, player_total_time))

            for player_idx, player_served in enumerate(self.served):
                self.logger.info("{} final bowl {}".format(self.player_names[player_idx], player_served))

            for player_idx, player_timeout_count in enumerate(self.timeout_count):
                if player_timeout_count > 0:
                    self.logger.info("{} timed out {} times".format(self.player_names[player_idx], player_timeout_count))

            for player_idx, player_error_count in enumerate(self.error_count):
                if player_error_count > 0:
                    self.logger.info("{} had exceptions {} times".format(self.player_names[player_idx], player_error_count))

            for player_idx, score in enumerate(self.player_scores):
                self.logger.info("{} individual score: {}".format(self.player_names[player_idx], score))
            group_score = np.mean(self.player_scores)
            final_scores = []
            self.logger.info("Average group score for all players: {}".format(group_score))
            for player_idx, score in enumerate(self.player_scores):
                other_player_scores = np.copy(self.player_scores)
                other_player_scores = np.delete(other_player_scores, player_idx)
                if other_player_scores.size == 0:
                    final_scores.append(score)
                else:
                    final_scores.append(np.mean([score, np.mean(other_player_scores)]))
                self.logger.info("{} final score: {}".format(self.player_names[player_idx], final_scores[player_idx]))

            self.final_scores = np.array(final_scores)

            winner_list_idx = np.argwhere(self.final_scores == np.amax(self.final_scores))
            self.winner_list = [self.player_names[i[0]] for i in winner_list_idx]

            self.logger.info("Winner{}: {}".format("s" if len(self.winner_list) > 1 else "", ", ".join(self.winner_list)))
            if self.use_gui:
                self.ice_cream_app.set_label_text("Winner{}: {}".format("s" if len(self.winner_list) > 1 else "", ", ".join(self.winner_list)), label_num=1)

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
        if self.use_gui:
            self.ice_cream_app.set_label_text("", label_num=1)
        self.logger.debug("Next turn {}".format(self.player_names[self.next_player]))
        if self.use_gui:
            self.ice_cream_app.set_label_text("{}, Next turn {}".format(self.ice_cream_app.get_label_text(), self.player_names[self.next_player]))

    def set_app(self, ice_cream_app):
        if self.use_gui:
            self.ice_cream_app = ice_cream_app
        else:
            self.logger.debug("No GUI flag specified, skipping setting app")

    def is_game_ended(self):
        return np.amin(self.turns_received) >= self.total_turn_per_player

    def play_all(self):
        if not self.is_game_ended():
            self.__log("Playing all turns")
            while not self.is_game_ended():
                self.play(run_stepwise=False, do_update=False)
            if self.use_gui:
                self.ice_cream_app.update_score_table()
                self.ice_cream_app.update_table()
            self.__game_end()

    def play(self, run_stepwise=False, do_update=True):
        if not self.processing_turn:
            if not self.is_game_ended():
                if np.amin(self.turns_received) < self.turns_received[self.next_player]:
                    self.logger.debug("Can't pass to the {}, as other player(s) with less helpings exist".format(self.player_names[self.next_player]))
                    self.next_player = self.__assign_next_player()
                    self.logger.debug("Assigned new player {}".format(self.player_names[self.next_player]))

                self.__log("Current turn {}".format(self.player_names[self.next_player]))
                self.__log("Scooped (f,p)", label_num=1)

                self.processing_turn = True
                self.served_this_turn = []
                self.turns_received[self.next_player] += 1
                self.time_taken[self.next_player].append([])

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
        
        if self.is_game_ended() and not self.end_message_printed:
            self.__game_end()

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
            try:
                if self.use_timeout:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(constants.timeout)
                try:
                    start_time = time.time()
                    action_values_dict = player.serve(top_layer, curr_level, player_idx, self.get_flavors, self.get_player_count, self.get_served, self.get_turns_received)
                    if self.use_timeout:
                        signal.alarm(0)      # Clear alarm
                except TimeoutException:
                    self.logger.error("Timeout {} since {:.3f}s reached.".format(self.player_names[player_idx], constants.timeout))
                    action_values_dict = dict()
                    self.timeout_count[player_idx] += 1
                step_time = time.time() - start_time
                self.time_taken[player_idx][-1].append(step_time)
            except Exception as e:
                self.logger.error(e, exc_info=True)
                action_values_dict = dict()
                self.error_count[player_idx] += 1
            is_valid_action = self.__check_action(action_values_dict)
            if is_valid_action:
                action = action_values_dict["action"]
                values = action_values_dict["values"]
                self.logger.debug("Received action: {} from {} in {:.3f}s".format(action_values_dict, self.player_names[player_idx], step_time))
                if self.use_gui:
                    self.ice_cream_app.set_label_text("{}, {}".format(self.ice_cream_app.get_label_text(), action_values_dict))

                if action == "scoop":
                    i, j = values
                    if not(i >= 0 and i < self.l-1 and j >= 0 and j < self.w-1):
                        self.logger.debug("Received out of bounds scoop position {}".format((i, j)))
                        pass_next = True
                    elif len(self.ice_cream_container.scoop(i, j, dry_run=True)) <= 0:
                        self.logger.debug("Received empty scooping position, passing to next player")
                        pass_next = True
                    elif len(self.ice_cream_container.scoop(i, j, dry_run=True)) + len(self.served_this_turn) <= self.max_allowed_per_turn:
                        scooped_items = self.ice_cream_container.scoop(i, j, dry_run=False)
                        for flavor in scooped_items:
                            self.served[player_idx][flavor] += 1
                            self.player_scores[player_idx] += len(self.flavors) - self.__get_flavor_preference(player_idx, flavor) + 1
                        scooped_items_preference = [(flavor, self.__get_flavor_preference(player_idx, flavor)) for flavor in scooped_items]

                        self.served_this_turn.extend(scooped_items)
                        self.logger.debug("Scooped (f,p): {} i.e. {} unit{}. So far scooped {} unit{} in this turn".format(scooped_items_preference, len(scooped_items), "s" if len(scooped_items) > 1 else "", len(self.served_this_turn), "s" if len(self.served_this_turn) > 1 else ""))
                        if self.use_gui:
                            self.ice_cream_app.set_label_text("{}, {}".format(self.ice_cream_app.get_label_text(label_num=1), scooped_items_preference), label_num=1)
                    else:
                        self.logger.debug("Scooping limit exceeded, passing to next player")
                        pass_next = True
                elif action == "pass":
                    if values < 0 or values >= len(self.players):
                        self.logger.debug("Next player idx {} is out of bounds".format(values))
                    elif values == player_idx:
                        if np.all(self.turns_received == self.turns_received[player_idx]):
                            # If turns received by all players is same as current player, then the current player is the last player in the turn and allowed to pass to themself
                            next_player = values
                        else:
                            self.logger.debug("Can't ask to pass to yourself, unless you are last in the turn")
                    else:
                        next_player = values
                    pass_next = True
            else:
                self.logger.debug("Received invalid action from player {}, passing to next player.".format(self.player_names[player_idx]))
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

    def get_current_player(self):
        return self.player_names[self.next_player]


class IceCreamApp(App):
    def __init__(self, *args):
        res_path = os.path.join(root_dir, 'res')
        super(IceCreamApp, self).__init__(*args, static_file_path={'res': res_path})

    def main(self, *userdata):
        self.ice_cream_game, start_automatic = userdata
        self.ice_cream_game.set_app(self)

        mainContainer = gui.Container(style={'width': '100%', 'display': 'block', 'overflow': 'auto', 'text-align': 'center'})
        mainContainer.style['justify-content'] = 'center'
        mainContainer.style['align-items'] = 'center'

        bt_hbox = gui.HBox(width="40%", style={'text-align': 'center', 'margin': 'auto'})
        play_step_bt = gui.Button("Play Step")
        play_turn_bt = gui.Button("Play Turn")
        play_all_bt = gui.Button("Play All")

        self.automatic_play = gui.CheckBoxLabel("Play Automatically", checked=start_automatic)
        self.automatic_play.attributes["class"] = "checkbox"

        bt_hbox.append([play_step_bt, play_turn_bt, play_all_bt, self.automatic_play])

        play_step_bt.onclick.do(self.play_step_bt_press)
        play_turn_bt.onclick.do(self.play_turn_bt_press)
        play_all_bt.onclick.do(self.play_all_bt_press)
        mainContainer.append(bt_hbox)
        self.labels = []
        self.labels.append(gui.Label("Ice Cream: Ready to start"))
        self.labels.append(gui.Label(""))
        self.ice_cream_game.logger.debug("First turn {}".format(self.ice_cream_game.get_current_player()))
        self.set_label_text("First turn {}".format(self.ice_cream_game.get_current_player()))
        for label in self.labels:
            mainContainer.append(label)

        self.score_table = gui.TableWidget(2, len(self.ice_cream_game.players)+1, style={'margin': '5px auto'})
        self.update_score_table()

        for player_idx, _ in enumerate(self.ice_cream_game.player_scores):
            self.score_table.item_at(0, player_idx).set_style("padding:0 10px")
            self.score_table.item_at(1, player_idx).set_style("padding:0 10px")
        mainContainer.append(self.score_table)

        paths = ["yellow.png", "brown.png", "green.png", "strawberry.png",  "almond.png",  "cream.png", "blue.png", "pink.png", "gray.png", "red.png", "orange.png", "cyan.png"]
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

    def idle(self):
        if not self.ice_cream_game.is_game_ended():
            if self.automatic_play.get_value():
                self.ice_cream_game.play(run_stepwise=True)
        else:
            self.automatic_play.set_value(False)

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

    def set_label_text(self, text, label_num=0):
        self.labels[label_num].set_text(text)

    def get_label_text(self, label_num=0):
        return self.labels[label_num].get_text()


class IceCreamContainer:
    def __init__(self, rng, logger, num_flavors=constants.default_num_flavor_choice) -> None:
        self.rng = rng
        self.logger = logger
        if num_flavors not in constants.num_flavor_choices:
            self.logger.debug("Num flavors {} is not in allowed values, using {} flavors".format(num_flavors, constants.default_num_flavor_choice))
            num_flavors = constants.default_num_flavor_choice
        self.logger.info("Generating ice cream with {} flavors".format(num_flavors))
        self.flavors = list(range(1, num_flavors+1))
        self.l = constants.length  # cols
        self.w = constants.width  # rows
        self.h = constants.height  # height
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
    num_flavor_choices = constants.num_flavor_choices
    num_flavor_choices.append(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--automatic", action="store_true", help="Start playing automatically in GUI mode")
    parser.add_argument("--seed", "-s", type=int, default=2021, help="Seed used by random number generator, specify 0 to use no seed and have different random behavior on each launch")
    parser.add_argument("--flavors", "-f", type=int, default=constants.default_num_flavor_choice, choices=num_flavor_choices, help="Number of flavors, specify 0 to use random number of flavors")
    parser.add_argument("--port", type=int, default=8080, help="Port to start")
    parser.add_argument("--address", "-a", type=str, default="127.0.0.1", help="Address")
    parser.add_argument("--no_browser", "-nb", action="store_true", help="Disable browser launching in GUI mode")
    parser.add_argument("--no_gui", "-ng", action="store_true", help="Disable GUI")
    parser.add_argument("--log_path", default="log", help="Directory path to dump log files, filepath if disable_logging is false")
    parser.add_argument("--disable_timeout", "-time", action="store_true", help="Disable Timeout in non GUI mode")
    parser.add_argument("--disable_logging", action="store_true", help="Disable Logging, log_path becomes path to file")
    parser.add_argument("--players", "-p", default=["1", "2", "3", "4", "5", "7", "8", "9", "10"], nargs="+", help="List of players space separated")
    args = parser.parse_args()
    player_list = tuple(args.players)
    del args.players

    if args.disable_logging:
        args.log_path = "results.log"

    ice_cream_game = IceCreamGame(player_list, args)
    if not ice_cream_game.use_gui:
        ice_cream_game.play_all()
        result = ice_cream_game.get_state()
