"""Real-time terminal dashboard for FlySoul combat telemetry."""

from __future__ import annotations

import sys
from typing import Dict, Any

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import ProgressBar
from rich.table import Table
from rich.text import Text
from rich.live import Live


class FlySoulDashboard:
    """Renders real-time bio-connectome and combat telemetry to the terminal."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console(legacy_windows=False)
        self.live: Live | None = None

    def create_layout(
        self,
        episode: int,
        step: int,
        player_hp_pct: float,
        player_sp_pct: float,
        boss_hp_pct: float,
        distance: float,
        boss_state: str,
        active_neurons: int,
        total_neurons: int,
        dopamine: float,
        action_name: str,
        motor_rates: Dict[str, float],
        cumulative_reward: float,
    ) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )
        layout["main"].split_row(
            Layout(name="combat", ratio=1),
            Layout(name="neural", ratio=1),
        )

        # Header
        header_text = Text(
            f"⚔️ FLYSOUL: MaleCNS v1.0 Connectome × Dark Souls III [Episode #{episode} | Step {step}]",
            style="bold cyan on dark_blue",
            justify="center",
        )
        layout["header"].update(Panel(header_text, style="blue"))

        # Combat Panel (Left)
        combat_table = Table(show_header=False, box=None, expand=True)

        # Boss HP
        boss_bar = "█" * int(boss_hp_pct * 30) + "░" * (30 - int(boss_hp_pct * 30))
        boss_style = "bold red" if boss_hp_pct < 0.3 else "bold yellow"
        combat_table.add_row(Text("👹 IUDEX GUNDYR", style="bold red"))
        combat_table.add_row(Text(f"HP [{boss_bar}] {boss_hp_pct*100:.1f}%", style=boss_style))
        combat_table.add_row(Text(f"State: {boss_state.upper()} | Distance: {distance:.1f}m", style="italic white"))
        combat_table.add_row(Text(""))

        # Player HP & SP
        player_bar = "█" * int(player_hp_pct * 30) + "░" * (30 - int(player_hp_pct * 30))
        player_style = "bold green" if player_hp_pct > 0.4 else "bold red"
        sp_bar = "█" * int(player_sp_pct * 20) + "░" * (20 - int(player_sp_pct * 20))

        combat_table.add_row(Text("🪰 FRUIT FLY (MaleCNS)", style="bold green"))
        combat_table.add_row(Text(f"HP [{player_bar}] {player_hp_pct*100:.1f}%", style=player_style))
        combat_table.add_row(Text(f"SP [{sp_bar}] {player_sp_pct*100:.1f}%", style="cyan"))
        combat_table.add_row(Text(""))
        combat_table.add_row(Text(f"⚡ Current Action: {action_name.upper()}", style="bold magenta"))

        combat_panel = Panel(combat_table, title="[bold red]Arena Status[/bold red]", border_style="red")
        layout["combat"].update(combat_panel)

        # Neural Panel (Right)
        neural_table = Table(show_header=False, box=None, expand=True)

        # Spiking metrics
        spike_pct = (active_neurons / max(1, total_neurons)) * 100.0
        neural_table.add_row(Text(f"🧠 Total Neurons: {total_neurons:,}"))
        neural_table.add_row(Text(f"⚡ Active Spiking: {active_neurons} ({spike_pct:.1f}%)", style="bold green"))

        # Dopamine state
        if dopamine < -0.05:
            dopa_text = Text(f"🔴 PPL101 Aversive Burst ({dopamine:.2f}) -> LTD", style="bold red")
        elif dopamine > 0.05:
            dopa_text = Text(f"🟢 PAM Reward Burst (+{dopamine:.2f}) -> LTP", style="bold green")
        else:
            dopa_text = Text("⚪ Basal Homeostasis (0.00)", style="dim")
        neural_table.add_row(Text("Dopamine Plasticity:"))
        neural_table.add_row(dopa_text)
        neural_table.add_row(Text(""))

        # Motor Firing Rates
        neural_table.add_row(Text("Descending Neuron (DN) Firing Pools:", style="bold white"))
        for pool, rate in motor_rates.items():
            rate_bar = "▓" * min(15, int(rate * 3))
            neural_table.add_row(Text(f"  {pool:<12}: {rate:>4.1f} Hz {rate_bar}", style="cyan"))

        neural_panel = Panel(neural_table, title="[bold cyan]Connectome Telemetry[/bold cyan]", border_style="cyan")
        layout["neural"].update(neural_panel)

        # Footer
        footer_text = Text(
            f"Cumulative Reward: {cumulative_reward:+.2f} | Synaptic Delay: 1.8ms | Integration: LIF Numba JIT",
            justify="center",
            style="italic yellow",
        )
        layout["footer"].update(Panel(footer_text, style="yellow"))

        return layout
