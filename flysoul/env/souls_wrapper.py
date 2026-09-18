"""Wrapper for the official SoulsGym Dark Souls III Gymnasium environment."""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym

from flysoul.env.mock_env import MockIudexEnv
from flysoul.env.mouse_input import enable_mouse_bindings

logger = logging.getLogger(__name__)


class ValidActionInfo(gym.Wrapper):
    """Publish the environment's currently executable actions in ``info``.

    SoulsGym silently discards any action the player cannot perform right now — during
    an attack recovery, a roll, or with no stamina — and that step is spent for nothing.
    Surfacing the mask lets the motor decoder read out only the pools the body can
    actually execute instead of issuing commands into a closed gate.
    """

    def _mask(self) -> list[int] | None:
        env = self.env.unwrapped
        getter = getattr(env, "current_valid_actions", None) or getattr(env, "valid_actions", None)
        if getter is None:
            return None
        try:
            return [int(a) for a in getter()]
        except Exception:  # pragma: no cover - the live game can race during transitions
            return None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        mask = self._mask()
        if mask is not None:
            info["valid_actions"] = mask
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        mask = self._mask()
        if mask is not None:
            info["valid_actions"] = mask
        return obs, reward, terminated, truncated, info


class LiveEnvUnavailable(RuntimeError):
    """The live game was asked for and could not be reached."""


def make_souls_env(
    use_mock: bool = False, skip_steps: bool = True, allow_fallback: bool = False,
    game_speed: float = 1.0,
) -> gym.Env:
    """Create either the live Dark Souls III SoulsGym environment or the mock simulator.

    Args:
        use_mock: If True, forces use of MockIudexEnv.
        skip_steps: Let SoulsGym advance through steps where the player is locked in an
            animation and only "do nothing" is executable. Every such step used to cost
            the agent a decision that the game then threw away.
        allow_fallback: Permit silently substituting the offline simulator when the live
            environment cannot be created. Off by default, and it should stay off.
        game_speed: Speed multiplier the game runs at during a step. The brain costs
            about a third of a game step to simulate, so up to 3x is feasible; SoulsGym's
            author trained at 3x. Resets are not affected.

    Raises:
        LiveEnvUnavailable: the live game was requested but could not be initialised.

    Returns:
        gym.Env instance.
    """
    if use_mock:
        logger.info("Initializing Standalone Mock Iudex Gundyr Environment...")
        return ValidActionInfo(MockIudexEnv())

    try:
        import soulsgym  # noqa: F401  (registers the environments)

        # Dark Souls III listens for attacks on the mouse, not on the keys soulsgym
        # sends. Route them to the buttons the game already responds to before the
        # environment builds its GameInput.
        routed = enable_mouse_bindings()
        logger.info(f"Input routed to the mouse: {routed}")
        print(f"OK Input routed to the mouse: {routed}", flush=True)

        logger.info("Initializing Live SoulsGym Iudex Environment (DarkSoulsIII.exe hook)...")
        try:
            env = gym.make("SoulsGymIudex-v0", skip_steps=skip_steps, game_speed=game_speed)
        except TypeError:
            # Older soulsgym releases do not accept these options.
            logger.warning("Installed soulsgym does not support skip_steps/game_speed; continuing without them.")
            env = gym.make("SoulsGymIudex-v0")
        return ValidActionInfo(env)

    except Exception as e:
        # Falling back here is worse than failing. The run carries on printing episode
        # results that look exactly like real ones, so a broken soulsgym install - a
        # syntax error in a patched module is enough - reads as hours of training
        # against Iudex when it was hours against the simulator.
        if not allow_fallback:
            raise LiveEnvUnavailable(
                f"--game was requested but SoulsGym could not be initialised: {e}"
            ) from e
        logger.warning(
            f"Could not initialize live SoulsGym ({e}). Falling back to MockIudexEnv simulation."
        )
        return ValidActionInfo(MockIudexEnv())


def is_mock(env: Any) -> bool:
    """True if this environment is the offline simulator rather than the live game."""
    return isinstance(getattr(env, "unwrapped", env), MockIudexEnv)


# Actions whose key must reach the game or the agent cannot fight at all.
_REQUIRED_KEYS = ("lightattack", "heavyattack", "parry", "lock_on")


def check_input_bindings(env, probe: str = "lightattack", attempts: int = 3) -> bool:
    """Press one key and report whether Dark Souls III reacted.

    Dark Souls III binds attacking to the mouse by default, and this input path only
    sends keyboard events. With the game's own bindings left alone, the agent walks and
    rolls perfectly - those are WASD and space, which do match - while every swing it
    asks for silently does nothing. The fight then always ends with the boss at full
    health, which reads as a policy that never learned to attack rather than as an
    input that never arrived.

    A single probe is not enough to conclude the binding is wrong: the window may not
    have focus, and the player may be mid-animation, dead or loading, in which case no
    input of any kind produces an animation. Focus the window, wait for the player to be
    idle, and probe a few times before declaring the key dead.

    Returns True if the probe produced a player animation.
    """
    import time

    game = getattr(env.unwrapped, "game", None)
    game_input = getattr(env.unwrapped, "_game_input", None)
    if game is None or game_input is None:
        return True  # Mock environment, or an install we cannot probe.

    try:
        game.game_speed = 1.0
        window = getattr(env.unwrapped, "_game_window", None) or getattr(game, "_game_window", None)
        if window is not None and not getattr(window, "focused", True):
            try:
                window.focus()
                time.sleep(0.5)
            except Exception:
                pass

        # Wait for a state in which an attack would animate at all.
        deadline = time.time() + 15.0
        while time.time() < deadline:
            game.clear_cache()
            if game.player_animation == "Idle" and game.player_hp > 0:
                break
            time.sleep(0.4)
        else:
            logger.warning(
                "Player never reached an idle state; skipping the attack input check."
            )
            return True  # Cannot tell; do not block the run on an inconclusive probe.

        for attempt in range(attempts):
            game_input.reset()
            time.sleep(0.3)
            game.clear_cache()
            before = game.player_animation
            game_input.single_action(probe, 0.15)
            probe_deadline = time.time() + 1.6
            while time.time() < probe_deadline:
                game.clear_cache()
                if game.player_animation != before:
                    game_input.reset()
                    return True
                time.sleep(0.05)
            game_input.reset()
            if attempt + 1 < attempts:
                logger.warning(f"Attack probe {attempt + 1}/{attempts} produced nothing; retrying.")
                time.sleep(1.0)
        return False
    except Exception as exc:
        logger.warning(f"Attack input check could not run ({exc}); continuing.")
        return True  # Never block a run on a probe that itself failed.


BINDING_HELP = """Dark Souls III is not bound to the keys SoulsGym sends.

Open the game: Settings -> Key Bindings, and set exactly these:

    Attack                l          Movement (already correct by default):
    Strong Attack         h              forward w   back s   left a   right d
    Parry / Left Attack   m              roll/backstep  space
    Lock On               q
    Event Action          e          Camera:
                                         up j   down k   left o   right p

The game binds attacking to the mouse out of the box, and this input path sends
keyboard events only - so an unbound attack key means the agent can walk, roll and
approach perfectly while never once landing a hit.

Verify with:  python scripts/check_input.py
"""
