"""Interfaz Jade 2.0: tablero SVG, reloj de práctica y controles táctiles.

No cambia la base de frecuencias ni entrena con las jugadas del usuario.
El constructor también funciona sin Colab para verificar la lógica y el HTML.
"""
import html
import json
import math
import os
import tempfile
import time
import uuid
from pathlib import Path

import chess
import chess.pgn
import chess.svg
from IPython.display import HTML, JSON, display


class JadeBoard:
    VERSION = "2.2.1"
    PROFILES = (1200, 1300, 1400, 1500, 1600)

    def __init__(self, jade, game_path=None, previous=None, now=None, sleeper=None):
        self.jade = jade
        self.board = chess.Board()
        self.human, self.orientation = chess.WHITE, chess.WHITE
        self.profile, self.beta = 1300, 1.0
        self.selected, self.pending = None, []
        self.finished, self.busy, self.closed, self.started = False, False, False, False
        self.result, self.message = "*", "Elige tu color y pulsa Jugar."
        self.last_details = None
        self.now, self.sleep = now or time.monotonic, sleeper or time.sleep
        self.initial_seconds, self.increment = 600., 0.
        self.remaining = [600., 600.]  # Índices: negras=0, blancas=1.
        self.paused, self.clock_started = False, None
        self.clock_history, self.move_clocks = [], []
        self.maximum, self.pace = 15, 1.
        self.revision = 0
        self.callback_name = "jade.ui." + uuid.uuid4().hex
        self.game_path = Path(game_path) if game_path is not None else None
        self.save_note = "La partida solo se conserva en esta sesión."
        self.load_error = None
        if previous is not None:
            if hasattr(previous, "settle_clock"):
                previous.settle_clock()
            # Conserva el historial, también al actualizar desde el tablero 1.0.
            self.board = previous.board.copy(stack=True)
            self.human = bool(previous.human)
            self.orientation = getattr(previous, "orientation", self.human)
            self.profile, self.beta = previous.profile, previous.beta
            self.finished, self.result = previous.finished, previous.result
            self.last_details = previous.last_details
            self.started = getattr(previous, "started", True)
            if hasattr(previous, "initial_seconds"):
                self.initial_seconds, self.increment = previous.initial_seconds, previous.increment
                self.remaining = list(previous.remaining)
                self.clock_history = [list(x) for x in previous.clock_history]
                self.move_clocks = list(previous.move_clocks)
                self.maximum, self.pace = previous.maximum, previous.pace
                self.paused = bool(self.started and not self.finished)
            else:
                # A una partida de 1.0/1.1 no se le inventa tiempo ya consumido.
                self.initial_seconds, self.remaining = 0., [0., 0.]
                self.clock_history = [[0., 0.] for _ in self.board.move_stack]
                self.move_clocks = [None for _ in self.board.move_stack]
            self.load_error = getattr(previous, "load_error", None)
            if self.load_error:
                self.message = self.load_error
                self.save_note = "El archivo original se conserva."
            else:
                self.message = "Partida conservada. Pulsa Continuar reloj." if self.paused else "Interfaz actualizada. Tu partida se conserva."
                self.save_game()
        elif self.game_path is not None and self.game_path.exists():
            try:
                self.restore_game(json.loads(self.game_path.read_text(encoding="utf-8")))
                self.message = "Partida recuperada en pausa. Pulsa Continuar reloj." if self.paused else "Partida recuperada. Puedes continuar."
                self.save_note = "Partida recuperada de la copia guardada."
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.load_error = "No se pudo recuperar la partida: " + str(exc)
                self.message = self.load_error
                self.save_note = "El archivo original se conserva."

    def restore_game(self, data):
        if data["schema"] not in (1, 2) or data["rhythm"] != self.jade.memory.rhythm:
            raise ValueError("La copia tiene otro formato o ritmo.")
        board = chess.Board(data["root_fen"])
        if not board.is_valid():
            raise ValueError("Posición inicial no válida.")
        for uci in data["moves"]:
            board.push_uci(uci)
        profile, beta = int(data["profile"]), float(data["beta"])
        if profile not in self.PROFILES or not math.isfinite(beta) or not 0 <= beta <= 2:
            raise ValueError("Ajustes de partida no válidos.")
        if data["result"] not in ("*", "1-0", "0-1", "1/2-1/2"):
            raise ValueError("Resultado no válido.")
        self.board, self.profile, self.beta = board, profile, beta
        self.human = bool(data["human"])
        self.orientation = bool(data.get("orientation", self.human))
        self.started, self.finished = bool(data["started"]), bool(data["finished"])
        self.result = data["result"]
        self.last_details = data.get("last_details")
        clock = data.get("clock", {})
        self.initial_seconds = float(clock.get("initial", 0))
        self.increment = float(clock.get("increment", 0))
        self.remaining = [float(x) for x in clock.get("remaining", [0., 0.])]
        if len(self.remaining) != 2 or any(not math.isfinite(x) or x < 0 for x in self.remaining):
            raise ValueError("Reloj guardado no válido.")
        self.clock_history = clock.get("history", [[0., 0.] for _ in board.move_stack])
        self.move_clocks = clock.get("moves", [None for _ in board.move_stack])
        if len(self.clock_history) != len(board.move_stack) or len(self.move_clocks) != len(board.move_stack):
            raise ValueError("Historial del reloj incompleto.")
        self.maximum, self.pace = data.get("maximum", 15), float(data.get("pace", 1.))
        if self.maximum not in (12, 15) or self.pace not in (.5, 1., 1.5):
            raise ValueError("Configuración del selector no válida.")
        self.paused = bool(self.initial_seconds and self.started and not self.finished)
        self.clock_started = None
        self.jade.new_game()
        self.update_outcome()

    def game_data(self):
        return dict(schema=2, rhythm=self.jade.memory.rhythm,
                    root_fen=self.board.root().fen(),
                    moves=[move.uci() for move in self.board.move_stack],
                    human=self.human, orientation=self.orientation, profile=self.profile,
                    beta=self.beta, started=self.started, finished=self.finished,
                    result=self.result, last_details=self.last_details,
                    maximum=self.maximum, pace=self.pace,
                    clock=dict(initial=self.initial_seconds, increment=self.increment,
                               remaining=self.remaining, history=self.clock_history, moves=self.move_clocks))

    def clock_running(self):
        return bool(self.initial_seconds and self.started and not self.finished and not self.paused and not self.closed)

    def settle_clock(self):
        if not self.clock_running() or self.clock_started is None:
            return
        stamp = self.now()
        side = int(self.board.turn)
        self.remaining[side] = max(0., self.remaining[side] - max(0., stamp - self.clock_started))
        self.clock_started = stamp
        if self.remaining[side] <= 0:
            opponent = not self.board.turn
            self.finished = True
            self.result = "1/2-1/2" if self.board.has_insufficient_material(opponent) else ("1-0" if opponent else "0-1")
            self.message = ("Tu tiempo se ha agotado" if self.board.turn == self.human else "Jade se ha quedado sin tiempo") + " · " + self.result
            self.clock_started = None
            self.pending = []
            self.save_game()

    def clock_view(self):
        values = list(self.remaining)
        if self.clock_running() and self.clock_started is not None:
            side = int(self.board.turn)
            values[side] = max(0., values[side] - max(0., self.now() - self.clock_started))
        return values

    def commit_move(self, move):
        self.settle_clock()
        if self.finished:
            return False
        self.clock_history.append(list(self.remaining))
        mover = int(self.board.turn)
        if self.initial_seconds:
            self.remaining[mover] += self.increment
        self.move_clocks.append(self.remaining[mover] if self.initial_seconds else None)
        self.board.push(move)
        self.clock_started = self.now() if self.clock_running() else None
        return True

    def save_game(self):
        if self.game_path is None:
            return
        temporary = None
        try:
            self.game_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".part",
                                             dir=self.game_path.parent, delete=False) as out:
                temporary = Path(out.name)
                json.dump(self.game_data(), out, ensure_ascii=False)
                out.flush()
                os.fsync(out.fileno())
            os.replace(temporary, self.game_path)
            place = "Drive" if str(self.game_path).startswith("/content/drive/") else "el almacenamiento configurado"
            self.save_note = "Partida guardada en " + place + "."
        except OSError as exc:
            self.save_note = "No se pudo guardar la partida. Descarga el PGN antes de salir. " + str(exc)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def archive_game(self):
        """Al empezar otra, conserva la anterior como PGN si tiene movimientos."""
        if self.game_path is not None and self.board.move_stack:
            path = self.game_path.parent / ("partida_" + uuid.uuid4().hex + ".pgn")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.pgn(), encoding="utf-8")

    def update_outcome(self):
        outcome = self.board.outcome()
        if outcome is not None:
            self.finished, self.result = True, outcome.result()
            reason = {"CHECKMATE": "Jaque mate", "STALEMATE": "Tablas por ahogado",
                      "INSUFFICIENT_MATERIAL": "Tablas por material insuficiente",
                      "SEVENTYFIVE_MOVES": "Tablas por 75 movimientos",
                      "FIVEFOLD_REPETITION": "Tablas por repetición"}
            self.message = reason.get(outcome.termination.name, "Partida terminada") + " · " + self.result

    def pgn(self):
        game = chess.pgn.Game.from_board(self.board)
        game.headers.update({"Event": "Práctica con Jade", "White": "Tú" if self.human else "Jade",
                             "Black": "Jade" if self.human else "Tú", "Result": self.result,
                             "JadeProfile": str(self.profile),
                             "TimeControl": f"{int(self.initial_seconds)}+{int(self.increment)}" if self.initial_seconds else "-"})
        for node, clock in zip(game.mainline(), self.move_clocks):
            if clock is not None:
                node.set_clock(clock)
        return str(game)

    def svg(self):
        targets = {move.to_square for move in self.board.legal_moves if move.from_square == self.selected}
        svg = chess.svg.board(
            self.board, orientation=self.orientation, coordinates=False,
            lastmove=self.board.peek() if self.board.move_stack else None,
            check=self.board.king(self.board.turn) if self.board.is_check() else None,
            fill={self.selected: "#f5cc69"} if self.selected is not None else {},
            colors={"square light": "#edf1df", "square dark": "#739889",
                    "square light lastmove": "#d9dc8a", "square dark lastmove": "#b7c46c"})
        # El SVG sin coordenadas usa casillas de 45 unidades. Las piezas proceden
        # de python-chess; estas marcas solo indican los destinos legales.
        dots = []
        for square in targets:
            file, rank = chess.square_file(square), chess.square_rank(square)
            col = file if self.orientation else 7 - file
            row = 7 - rank if self.orientation else rank
            x, y = 45 * col + 22.5, 45 * row + 22.5
            if self.board.piece_at(square):
                dots.append(f'<circle cx="{x}" cy="{y}" r="19" fill="none" stroke="#153f32" stroke-width="3" opacity=".7"/>')
            else:
                dots.append(f'<circle cx="{x}" cy="{y}" r="6" fill="#153f32" opacity=".45"/>')
        return svg.replace("</svg>", "".join(dots) + "</svg>")

    def state(self):
        self.update_outcome()
        files = list("abcdefgh") if self.orientation else list("hgfedcba")
        ranks = list(range(8, 0, -1)) if self.orientation else list(range(1, 9))
        names = {chess.PAWN: "peón", chess.KNIGHT: "caballo", chess.BISHOP: "alfil",
                 chess.ROOK: "torre", chess.QUEEN: "dama", chess.KING: "rey"}
        targets = {m.to_square for m in self.board.legal_moves if m.from_square == self.selected}
        squares = []
        for rank in ranks:
            for file in files:
                name = file + str(rank)
                sq = chess.parse_square(name)
                piece = self.board.piece_at(sq)
                label = name + (": " + names[piece.piece_type] + (" blanco" if piece.color else " negro") if piece else ": vacía")
                squares.append(dict(name=name, label=label, selected=sq == self.selected, target=sq in targets))
        replay, history = self.board.root(), []
        for move in self.board.move_stack:
            number, white = replay.fullmove_number, replay.turn
            san = replay.san(move)
            if white or not history:
                history.append([number, san if white else "…", ""])
            if not white:
                history[-1][2] = san
            replay.push(move)
        ai_pending = self.started and not self.finished and not self.paused and self.board.turn != self.human and not self.closed
        return dict(revision=self.revision, svg=self.svg(), squares=squares, files=files, ranks=ranks,
                    message=self.message, started=self.started, finished=self.finished, closed=self.closed,
                    human=self.human, orientation=self.orientation, profile=self.profile, beta=self.beta, rhythm=self.jade.memory.rhythm,
                    result=self.result, history=history, pgn=self.pgn(), plies=len(self.board.move_stack),
                    can_move=self.started and not self.finished and not self.closed and not self.paused and self.board.turn == self.human,
                    can_undo=bool(self.board.move_stack) and not (not self.human and len(self.board.move_stack) == 1),
                    can_draw=self.started and not self.finished and self.board.turn == self.human and self.board.can_claim_draw(),
                    ai_pending=ai_pending, promotion=bool(self.pending), save_note=self.save_note,
                    observations=(self.last_details or {}).get("candidate_observations", (self.last_details or {}).get("top5_observations")),
                    decision=self.last_details, maximum=self.maximum, pace=self.pace,
                    clock=dict(enabled=bool(self.initial_seconds), remaining=self.clock_view(),
                               active=int(self.board.turn) if self.clock_running() else None,
                               paused=self.paused, initial=self.initial_seconds, increment=self.increment),
                    turn=("Partida terminada" if self.finished else "En pausa" if self.paused else "Tu turno" if self.board.turn == self.human else "Turno de Jade"))

    def push_human(self, move):
        if move not in self.board.legal_moves:
            self.message = "Esa jugada no es legal. Selecciona un destino marcado."
            return
        self.message = "Has jugado " + self.board.san(move) + "."
        if not self.commit_move(move):
            return
        self.selected, self.pending = None, []
        self.update_outcome()
        self.save_game()  # Se guarda también antes de que Stockfish responda.

    def ai_turn(self):
        if not self.started or self.finished or self.paused or self.board.turn == self.human:
            return
        if self.board.can_claim_draw():
            self.finished, self.result, self.message = True, "1/2-1/2", "Jade reclama tablas."
        else:
            try:
                began = self.now()
                seconds = self.clock_view()[int(self.board.turn)] if self.initial_seconds else None
                move, details = self.jade.choose(self.board, self.profile, self.beta,
                                                 remaining=seconds, maximum=self.maximum, pace=self.pace)
                if move not in self.board.legal_moves:
                    raise ValueError("El motor devolvió una jugada no válida.")
                delay = max(0., details.get("reaction_seconds", 0.) - (self.now() - began))
                if self.initial_seconds:
                    delay = min(delay, max(0., self.clock_view()[int(self.board.turn)] * .5))
                if delay:
                    self.sleep(delay)
                self.settle_clock()
                if not self.finished:
                    self.message = "Jade juega " + self.board.san(move) + ". Te toca."
                    if self.commit_move(move):
                        details["elapsed_seconds"] = self.now() - began
                        self.last_details = details
                        self.update_outcome()
            except Exception as exc:
                self.message = "No se pudo calcular la respuesta. Pulsa Reintentar. Detalle: " + str(exc)
        self.save_game()

    def handle(self, action, payload=None, revision=None):
        """Un único punto de entrada. Python valida todas las jugadas y el turno."""
        if self.closed:
            return dict(self.state(), message="Este tablero se ha cerrado. Usa la salida más reciente de la celda 6.")
        if self.busy:
            return self.state()
        if revision is not None and int(revision) != self.revision:
            return dict(self.state(), message="Tablero sincronizado. Repite tu acción.")
        payload = payload or {}
        self.busy = True
        try:
            self.settle_clock()
            if action == "new":
                profile, beta = int(payload.get("profile", 1300)), float(payload.get("beta", 1))
                if profile not in self.PROFILES or not math.isfinite(beta) or not 0 <= beta <= 2:
                    raise ValueError("Elige uno de los perfiles disponibles y un peso entre 0 y 2.")
                initial, increment = map(int, str(payload.get("time_control", "600+0")).split("+"))
                maximum, pace = int(payload.get("maximum", 15)), float(payload.get("pace", 1.))
                if (initial, increment) not in ((0, 0), (180, 2), (300, 0), (600, 0), (600, 5), (900, 10)) or maximum not in (12, 15) or pace not in (.5, 1., 1.5):
                    raise ValueError("Selecciona un reloj y un ritmo de respuesta válidos.")
                # Un archivo ilegible se conserva antes de crear la nueva partida.
                if self.load_error and self.game_path and self.game_path.exists():
                    recovered = self.game_path.with_name("copia_no_leida_" + uuid.uuid4().hex + ".json")
                    recovered.write_bytes(self.game_path.read_bytes())
                self.archive_game()
                self.jade.new_game()
                self.board = chess.Board()
                self.human = payload.get("color", "white") == "white"
                self.orientation = self.human
                self.profile, self.beta = profile, beta
                self.initial_seconds, self.increment = float(initial), float(increment)
                self.remaining = [float(initial), float(initial)]
                self.clock_history, self.move_clocks = [], []
                self.maximum, self.pace, self.paused = maximum, pace, False
                self.started, self.finished, self.result = True, False, "*"
                self.selected, self.pending, self.last_details = None, [], None
                self.message, self.load_error = "Toca una pieza y después un destino marcado.", None
                self.clock_started = self.now() if initial else None
                self.save_game()
            elif action == "pause" and self.started and not self.finished:
                self.paused = not self.paused
                self.clock_started = self.now() if self.clock_running() else None
                self.message = "Reloj en pausa." if self.paused else "Partida reanudada."
                self.save_game()
            elif action == "tick":
                pass  # settle_clock ya comprueba la caída de bandera.
            elif action == "flip":
                self.orientation = not self.orientation
                if self.started:
                    self.save_game()
            elif action == "undo" and self.board.move_stack:
                if not self.human and len(self.board.move_stack) == 1:
                    self.message = "Todavía no has movido."
                else:
                    while self.board.move_stack:
                        self.board.pop()
                        if self.clock_history:
                            self.remaining = self.clock_history.pop()
                        if self.move_clocks:
                            self.move_clocks.pop()
                        if self.board.turn == self.human:
                            break
                    self.finished, self.result, self.selected = False, "*", None
                    self.pending, self.last_details = [], None
                    self.message = "Turno deshecho. Puedes elegir otra jugada."
                    self.clock_started = self.now() if self.clock_running() else None
                    self.save_game()
            elif action == "resign" and self.started and not self.finished:
                self.finished, self.result = True, "0-1" if self.human else "1-0"
                self.message, self.pending = "Te has rendido. Gana Jade.", []
                self.save_game()
            elif action == "draw" and self.started and not self.finished:
                if self.board.turn == self.human and self.board.can_claim_draw():
                    self.finished, self.result, self.message = True, "1/2-1/2", "Tablas reclamadas."
                    self.pending = []
                    self.save_game()
                else:
                    self.message = "Esta posición todavía no permite reclamar tablas."
            elif action == "ai":
                self.ai_turn()
            elif action == "cancel_promotion":
                self.pending = []
                self.message = "Promoción cancelada. Puedes elegir otro movimiento."
            elif self.started and not self.finished and not self.paused and self.board.turn == self.human:
                if action == "square":
                    sq = chess.parse_square(payload["square"])
                    piece = self.board.piece_at(sq)
                    if piece and piece.color == self.human:
                        self.selected = None if self.selected == sq else sq
                        self.pending = []
                        self.message = "Elige un destino marcado." if self.selected is not None else "Toca una pieza."
                    elif self.selected is not None:
                        moves = [m for m in self.board.legal_moves if m.from_square == self.selected and m.to_square == sq]
                        if any(m.promotion for m in moves):
                            self.pending = moves
                            self.message = "Elige la pieza para promocionar."
                        elif moves:
                            self.push_human(moves[0])
                        else:
                            self.message = "Toca uno de los destinos marcados."
                elif action == "promote":
                    symbol = payload.get("piece", "q")
                    move = next((m for m in self.pending if chess.piece_symbol(m.promotion) == symbol), None)
                    if move is not None:
                        self.push_human(move)
                elif action == "uci":
                    self.push_human(chess.Move.from_uci(str(payload.get("uci", "")).strip().lower()))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            self.message = "No se pudo realizar la acción: " + str(exc)
        finally:
            self.busy = False
            self.revision += 1
        return self.state()

    def callback(self, action, payload=None, revision=None):
        return JSON(self.handle(action, payload, revision))

    def html(self):
        # Evita cerrar el bloque script al incorporar textos del estado.
        initial = json.dumps(self.state(), ensure_ascii=False).replace("<", "\\u003c")
        return (JADE_HTML.replace("__JADE_CALLBACK__", self.callback_name)
                .replace("__JADE_INITIAL__", initial))

    def show(self):
        from google.colab import output
        output.register_callback(self.callback_name, self.callback)
        display(HTML(self.html()))

    def close(self):
        self.settle_clock()
        if self.started:
            self.paused, self.clock_started = True, None
            self.save_game()
        self.closed = True


JADE_HTML = r'''
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<style>
html,body{margin:0;padding:0;width:100%;min-width:0;background:transparent}
.jade,.jade *{box-sizing:border-box}
.jade{color-scheme:light;width:100%;max-width:460px;min-width:0;margin:0 auto;padding:12px;
  background:#f7f8f3;color:#183a30;border:1px solid #dce4d8;border-radius:18px;
  font:15px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;overflow-wrap:anywhere}
.jade button,.jade select,.jade input,.jade textarea{font:inherit;max-width:100%;min-width:0;color:#183a30;color-scheme:light}
.jade button{border:1px solid #c7d4c8;border-radius:10px;background:#fff;padding:10px 12px;min-height:44px;
  cursor:pointer;touch-action:manipulation;-webkit-tap-highlight-color:transparent}
.jade button:disabled{cursor:default;opacity:.5}
.jade button:focus-visible,.jade summary:focus-visible,.jade select:focus-visible{outline:3px solid #216e55;outline-offset:2px}
.jade .brand{display:flex;justify-content:space-between;gap:8px;align-items:center;margin-bottom:12px}
.jade h1{font-size:29px;line-height:1.1;letter-spacing:-1px;margin:0}
.jade .muted{color:#52685b;font-size:12px}
.jade .tag{background:#e4ecdf;padding:5px 9px;border-radius:20px;font-size:12px;white-space:nowrap}
.jade .setup-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:10px 0}
.jade label{display:block;font-size:13px;font-weight:600}
.jade select,.jade input[type=text],.jade textarea{width:100%;background:#fff;border:1px solid #bbcbbb;border-radius:9px;padding:10px;margin-top:4px;min-height:44px}
.jade .primary{width:100%;background:#1b624b;border-color:#1b624b;color:white;font-weight:650}
.jade details{border-top:1px solid #dce4d8;padding:10px 0}
.jade summary{cursor:pointer;font-weight:600;min-height:32px;display:list-item;margin-left:18px}
.jade .advanced{padding:8px 0}
.jade input[type=range]{width:100%;accent-color:#1b624b}
.jade .status{background:#e9eee3;border-radius:10px;padding:9px 11px;margin:12px 0 8px}
.jade #turn{font-weight:750;display:block}
.jade #message{font-size:13px;display:block;min-height:19px}
.jade .player{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:7px 0;font-size:13px}
.jade .player b{font-weight:650}
.jade .clock{font:700 21px/1.1 ui-monospace,monospace;letter-spacing:1px;padding:7px 10px;border-radius:8px;background:#e4eadf;color:#3d5144;white-space:nowrap}
.jade .clock.active{background:#174d3b;color:#fff}
.jade .clock.low{background:#9d392e;color:#fff}
.jade .clock.paused{opacity:.65}
.jade .board-frame{display:grid;grid-template-columns:12px minmax(0,1fr);grid-template-rows:auto 16px;gap:2px;width:100%;min-width:0}
.jade .ranks{display:grid;grid-template-rows:repeat(8,1fr);font-size:10px;text-align:center;align-items:center;color:#52685b}
.jade .files{grid-column:2;display:grid;grid-template-columns:repeat(8,1fr);font-size:10px;text-align:center;color:#52685b}
.jade .board{position:relative;aspect-ratio:1;width:100%;min-width:0;border-radius:4px;overflow:hidden;background:#739889}
.jade #drawing{position:absolute;inset:0}
.jade #drawing svg{display:block;width:100%;height:100%;max-width:100%}
.jade .squares{position:absolute;inset:0;display:grid;grid-template-columns:repeat(8,minmax(0,1fr));grid-template-rows:repeat(8,minmax(0,1fr))}
.jade .squares button{display:block;width:100%;height:100%;min-height:0;padding:0;border:0;margin:0;border-radius:0;background:transparent!important;opacity:1!important}
.jade .squares button:focus-visible{outline:3px solid #17664b;outline-offset:-3px}
.jade .actions{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:10px 0}
.jade .actions button{font-size:13px;padding:8px}
.jade .danger{color:#9a3830;border-color:#e5b9b0}
.jade [hidden]{display:none!important}
.jade .promotion,.jade .confirm{background:#fff9e6;border:1px solid #e1c475;border-radius:10px;padding:10px;margin:10px 0}
.jade .promotions{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin:8px 0}
.jade .history{max-height:150px;overflow:auto;background:#fff;border-radius:8px;margin:6px 0}
.jade table{border-collapse:collapse;width:100%;font-size:14px;text-align:left;table-layout:fixed}
.jade th,.jade td{padding:6px 10px;border-bottom:1px solid #e8eee4}
.jade th:first-child,.jade td:first-child{width:40px;color:#617264}
.jade textarea{height:110px;font:12px/1.5 ui-monospace,monospace;resize:vertical}
.jade .footer{margin-top:8px;font-size:11px;color:#607264}
@media(max-width:280px){.jade{padding:8px;border-radius:12px;font-size:14px}.jade .setup-grid{grid-template-columns:1fr}.jade .tag{font-size:10px}.jade button{padding:8px}.jade .actions{gap:5px}}
</style>
<main class="jade" id="jade">
  <header class="brand"><div><h1>Jade</h1><div class="muted">Práctica con protección táctica</div></div><span class="tag">Jade 2.2.1</span></header>
  <details id="setup" open><summary>Preparar partida</summary>
    <div class="setup-grid">
      <label>Perfil de Jade<select id="profile"><option>1200</option><option selected>1300</option><option>1400</option><option>1500</option><option>1600</option></select></label>
      <label>Tú juegas con<select id="color"><option value="white">Blancas</option><option value="black">Negras</option></select></label>
      <label>Reloj<select id="time-control"><option value="600+0">10 minutos</option><option value="600+5">10 min + 5 s</option><option value="900+10">15 min + 10 s</option><option value="300+0">5 minutos</option><option value="180+2">3 min + 2 s</option><option value="0+0">Sin reloj</option></select></label>
      <label>Respuesta de Jade<select id="pace"><option value="1">Natural</option><option value="1.5">Pausada</option><option value="0.5">Ágil</option></select></label>
    </div>
    <details><summary>Ajuste de estilo</summary><div class="advanced">
      <label>Influencia humana: <output id="weight-label">1.00</output><input id="weight" type="range" min="0" max="2" step=".25" value="1"></label>
      <div class="muted">1 es el valor inicial. 0 usa solo el motor y la temperatura. Los perfiles son orientativos, todavía no son ELO medidos.</div>
      <label style="margin-top:8px">Máximo de opciones<select id="maximum"><option>15</option><option>12</option></select></label>
      <div class="muted">Las opciones bajan hasta 3 con poco tiempo. Las jugadas humanas pueden entrar aunque no estén entre las favoritas del motor. El reloj es de práctica; puedes pausarlo.</div>
    </div></details>
    <button id="new" class="primary">Jugar</button>
  </details>
  <div class="confirm" id="confirm" hidden><b id="confirm-text"></b><div class="actions"><button id="confirm-yes">Confirmar</button><button id="confirm-no">Cancelar</button></div></div>
  <div class="status" role="status" aria-live="polite"><span id="turn">Listo para jugar</span><span id="message">Toca Jugar para empezar.</span></div>
  <div class="player"><div><b id="opponent">Jade · perfil 1300</b><div id="rhythm" class="muted">Rapid</div></div><span class="clock" id="clock-top" aria-label="Reloj superior">10:00</span></div>
  <div class="board-frame">
    <div class="ranks" id="ranks" aria-hidden="true"></div>
    <div class="board" id="board"><div id="drawing" aria-hidden="true"></div><div class="squares" id="squares" role="group" aria-label="Tablero de ajedrez"></div></div>
    <div class="files" id="files" aria-hidden="true"></div>
  </div>
  <div class="player"><div><b id="you">Tú · blancas</b><div class="muted">Toca origen y destino</div></div><span class="clock" id="clock-bottom" aria-label="Reloj inferior">10:00</span></div>
  <div class="promotion" id="promotion" role="group" aria-label="Promocionar peón" hidden><b>¿En qué pieza lo conviertes?</b><div class="promotions">
    <button data-piece="q">Dama</button><button data-piece="r">Torre</button><button data-piece="b">Alfil</button><button data-piece="n">Caballo</button>
  </div><button id="cancel-promotion">Cancelar</button></div>
  <button id="retry" class="primary" hidden>Reintentar respuesta de Jade</button>
  <div class="actions"><button id="undo">Deshacer turno</button><button id="flip">Girar tablero</button></div>
  <button id="pause" style="width:100%;margin-bottom:8px">Pausar reloj</button>
  <details id="history-panel"><summary>Movimientos <span id="count"></span></summary><div id="history" class="history"></div></details>
  <details><summary>Opciones de partida</summary>
    <div class="actions"><button id="draw">Reclamar tablas</button><button id="resign" class="danger">Rendirme</button></div>
    <label>Jugada por texto (UCI)<input id="uci" type="text" placeholder="e2e4" autocomplete="off" autocapitalize="none" spellcheck="false"></label>
    <button id="send" style="width:100%;margin-top:6px">Jugar movimiento</button>
    <div class="actions"><button id="download">Descargar PGN</button><button id="copy">Copiar PGN</button></div>
    <textarea id="pgn" readonly aria-label="Partida en PGN"></textarea>
    <div id="evidence" class="muted"></div>
  </details>
  <div class="footer" id="save-note"></div>
  <div class="footer">Piezas: C. M. Burnett · SVG de python-chess.<br>Las partidas jugadas no se añaden a la memoria humana.</div>
</main>
<script>
(()=>{
  const root=document.getElementById('jade');
  const el=id=>root.querySelector('#'+id);
  const callback='__JADE_CALLBACK__';
  let state=__JADE_INITIAL__, waiting=false, confirmation=null, clockStamp=performance.now(), flagSent=false;
  const status=text=>{el('message').textContent=text};
  function fit(){try{google.colab.output.setIframeHeight(Math.ceil(root.getBoundingClientRect().height)+4,true)}catch(e){}}
  function lock(){
    root.querySelectorAll('button,select,input').forEach(x=>x.disabled=waiting||state.closed);
    root.querySelectorAll('.squares button').forEach(x=>x.disabled=waiting||!state.can_move||state.promotion);
    el('undo').disabled=waiting||state.closed||!state.can_undo;
    el('draw').disabled=waiting||state.closed||!state.can_draw;
    el('resign').disabled=waiting||state.closed||!state.started||state.finished;
    el('send').disabled=waiting||!state.can_move||state.promotion;
    el('pause').disabled=waiting||state.closed||!state.started||state.finished;
    el('board').setAttribute('aria-busy',waiting?'true':'false');
  }
  function render(s){
    state=s;clockStamp=performance.now();flagSent=false;
    el('drawing').innerHTML=s.svg;
    el('ranks').replaceChildren(...s.ranks.map(x=>{const e=document.createElement('span');e.textContent=x;return e}));
    el('files').replaceChildren(...s.files.map(x=>{const e=document.createElement('span');e.textContent=x;return e}));
    el('squares').replaceChildren(...s.squares.map(q=>{
      const b=document.createElement('button');b.type='button';b.dataset.square=q.name;
      b.setAttribute('aria-label',q.label+(q.target?', destino legal':''));b.setAttribute('aria-pressed',q.selected?'true':'false');
      b.addEventListener('click',()=>send('square',{square:q.name}));return b;
    }));
    el('turn').textContent=s.closed?'Tablero cerrado':s.started?s.turn:'Listo para jugar';
    status(s.message);
    const humanLabel='Tú · '+(s.human?'blancas':'negras'), jadeLabel='Jade · perfil '+s.profile;
    el('opponent').textContent=s.orientation===s.human?jadeLabel:humanLabel;
    el('rhythm').textContent=s.rhythm==='rapid'?'Datos de rápidas':'Datos de blitz';
    el('you').textContent=s.orientation===s.human?humanLabel:jadeLabel;
    el('pause').textContent=s.clock.paused?'Continuar reloj':'Pausar reloj';
    el('new').textContent=s.started?'Nueva partida':'Jugar';
    el('retry').hidden=!s.ai_pending;
    el('promotion').hidden=!s.promotion;
    el('count').textContent=s.history.length?'('+s.history.length+')':'';
    const table=document.createElement('table');
    const header=document.createElement('tr');
    ['N.º','Blancas','Negras'].forEach(t=>{const th=document.createElement('th');th.textContent=t;header.append(th)});
    table.append(header);
    s.history.forEach(row=>{const tr=document.createElement('tr');row.forEach(t=>{const td=document.createElement('td');td.textContent=t;tr.append(td)});table.append(tr)});
    el('history').replaceChildren(table);el('history').scrollTop=el('history').scrollHeight;
    el('pgn').value=s.pgn;
    el('save-note').textContent=s.save_note;
    const d=s.decision;
    el('evidence').textContent=d?.candidate_count?
      d.candidate_count+' opciones · '+Math.round(100*d.human_mix)+' % de mezcla humana · '+d.candidate_observations+' ejemplos entre esas opciones. Última respuesta: '+(d.elapsed_seconds||0).toFixed(1)+' s.'+
      (d.selected_human_count?' La jugada elegida aparece '+d.selected_human_count+' veces en este perfil.':'')+
      (d.neural_active?' Predictor humano activo.':''):
      s.observations===null?'':s.observations+' observaciones humanas en la última respuesta.';
    lock();clocks();fit();
  }
  function clocks(){
    const c=state.clock;if(!c)return;
    const values=[...c.remaining];
    if(c.active!==null)values[c.active]=Math.max(0,values[c.active]-(performance.now()-clockStamp)/1000);
    const bottom=state.orientation?1:0;
    [['clock-top',1-bottom],['clock-bottom',bottom]].forEach(([id,side])=>{
      const t=values[side];el(id).textContent=c.enabled?(t<10?t.toFixed(1):Math.floor(t/60)+':'+String(Math.floor(t%60)).padStart(2,'0')):'Sin reloj';
      el(id).className='clock'+(c.active===side?' active':'')+(c.enabled&&t<=10?' low':'')+(c.paused?' paused':'');
    });
    if(c.active!==null&&values[c.active]<=0&&!waiting&&!flagSent&&!state.finished){flagSent=true;send('tick');}
  }
  async function call(action,payload={}){
    const result=await google.colab.kernel.invokeFunction(callback,[action,payload,state.revision],{});
    let data=result.data?.['application/json'];
    if(typeof data==='string')data=JSON.parse(data);
    if(!data||!data.squares)throw new Error('No se recibió el estado del tablero.');
    render(data);return data;
  }
  async function send(action,payload={}){
    if(waiting||state.closed)return;
    waiting=true;lock();
    try{
      if(action==='ai'){el('turn').textContent='Jade está pensando…';status('Calculando su respuesta.');el('retry').hidden=true;}
      const s=await call(action,payload);
      if(action==='new'){el('setup').open=false;fit()}
      if(s.ai_pending&&['new','square','promote','uci','pause'].includes(action)){
        el('turn').textContent='Jade está pensando…';status('Calculando su respuesta.');el('retry').hidden=true;
        // Permite pintar tu movimiento antes de iniciar la búsqueda del motor.
        await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
        await call('ai');
      }
    }catch(err){
      status('No se pudo conectar. Comprueba «Conectar» en Colab y vuelve a abrir la celda 6. '+err.message);
    }finally{waiting=false;lock();fit()}
  }
  function confirm(text,action){confirmation=action;el('confirm-text').textContent=text;el('confirm').hidden=false;fit()}
  el('confirm-no').onclick=()=>{confirmation=null;el('confirm').hidden=true;fit()};
  el('confirm-yes').onclick=()=>{const fn=confirmation;confirmation=null;el('confirm').hidden=true;if(fn)fn()};
  el('new').onclick=()=>{
    const start=()=>send('new',{profile:Number(el('profile').value),color:el('color').value,beta:Number(el('weight').value),time_control:el('time-control').value,maximum:Number(el('maximum').value),pace:Number(el('pace').value)});
    if(state.plies&&!state.finished)confirm('¿Dejar esta partida y empezar otra?',start);else start();
  };
  el('weight').oninput=()=>el('weight-label').value=Number(el('weight').value).toFixed(2);
  el('undo').onclick=()=>send('undo');el('flip').onclick=()=>send('flip');el('draw').onclick=()=>send('draw');
  el('resign').onclick=()=>confirm('¿Quieres rendirte en esta partida?',()=>send('resign'));
  el('retry').onclick=()=>send('ai');
  el('pause').onclick=()=>send('pause');
  el('send').onclick=()=>{const uci=el('uci').value;el('uci').value='';send('uci',{uci})};
  el('uci').onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();el('send').click()}};
  root.querySelectorAll('[data-piece]').forEach(b=>b.onclick=()=>send('promote',{piece:b.dataset.piece}));
  el('cancel-promotion').onclick=()=>send('cancel_promotion');
  el('download').onclick=()=>{
    const url=URL.createObjectURL(new Blob([state.pgn],{type:'application/x-chess-pgn'}));
    const a=document.createElement('a');a.href=url;a.download='Jade_partida.pgn';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),10000);
  };
  el('copy').onclick=async()=>{
    try{await navigator.clipboard.writeText(state.pgn);status('PGN copiado.');}
    catch(e){el('pgn').focus();el('pgn').select();status('Texto seleccionado. Mantén pulsado y elige Copiar.');}
  };
  root.querySelectorAll('details').forEach(d=>d.addEventListener('toggle',fit));
  el('profile').value=String(state.profile);el('color').value=state.human?'white':'black';
  el('weight').value=state.beta;el('weight-label').value=Number(state.beta).toFixed(2);
  el('maximum').value=state.maximum;el('pace').value=state.pace;
  el('time-control').value=state.clock.initial+'+'+state.clock.increment;
  el('setup').open=!state.started;render(state);
  const ticker=setInterval(()=>{if(!root.isConnected)clearInterval(ticker);else clocks()},200);
  if(typeof ResizeObserver!=='undefined')new ResizeObserver(fit).observe(root);
  window.addEventListener('resize',fit);
})();
</script>
'''
