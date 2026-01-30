#!/usr/bin/env python3
"""
Script pour ouvrir le dashboard de visualisation dans le navigateur.
"""

import webbrowser
import os
from pathlib import Path
import time

def open_dashboard(dashboard_type="dynamic"):
    """
    Ouvre le dashboard dans le navigateur par défaut.
    Args:
        dashboard_type: Type de dashboard ("dynamic", "static", "text")
    """
    monitoring_dir = Path("monitoring")
    # Déterminer le fichier à ouvrir
    if dashboard_type == "dynamic":
        file_path = monitoring_dir / "dashboard_dynamic.html"
        print("🚀 Ouverture du dashboard dynamique (interactif)...")
    elif dashboard_type == "static":
        file_path = monitoring_dir / "dashboard_alerts.html"
        print("🚀 Ouverture du dashboard statique...")
    elif dashboard_type == "text":
        file_path = monitoring_dir / "dashboard_alerts.txt"
        print("📄 Affichage du dashboard texte...")
    else:
        print(f"❌ Type de dashboard inconnu: {dashboard_type}")
        return
    # Vérifier que le fichier existe
    if not file_path.exists():
        print(f"❌ Fichier non trouvé: {file_path}")
        print(f"💡 Générez d'abord le dashboard avec:")
        if dashboard_type == "dynamic":
            print("   python3 src/visualize_alerts.py")
        else:
            print("   python3 src/visualize_alerts.py")
        return
    # Afficher le contenu texte ou ouvrir dans le navigateur
    if dashboard_type == "text":
        with open(file_path, 'r', encoding='utf-8') as f:
            print("\n" + "="*80)
            print(f.read())
            print("="*80)
    else:
        abs_path = file_path.resolve()
        url = f"file://{abs_path}"
        print(f"📂 Fichier: {abs_path}")
        print(f"🌐 URL: {url}")
        print("⏳ Ouverture du navigateur...")
        webbrowser.open(url)
        print("✅ Dashboard ouvert dans le navigateur!")
        print("\n💡 Astuce: Si le navigateur ne s'ouvre pas automatiquement,")
        print(f"   copiez-collez cette URL dans votre navigateur: {url}")

def main():
    """Point d'entrée principal."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Ouvre le dashboard de monitoring dans le navigateur",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python3 open_dashboard.py                    # Dashboard dynamique (par défaut)
  python3 open_dashboard.py --type dynamic     # Dashboard interactif avec graphiques
  python3 open_dashboard.py --type static      # Dashboard HTML simple
  python3 open_dashboard.py --type text        # Dashboard texte dans le terminal
  # Raccourcis
  python3 open_dashboard.py -d                 # dynamic
  python3 open_dashboard.py -s                 # static
  python3 open_dashboard.py -t                 # text
        """
    )
    parser.add_argument(
        '--type', '-y',
        choices=['dynamic', 'static', 'text'],
        default='dynamic',
        help='Type de dashboard à ouvrir (défaut: dynamic)'
    )
    parser.add_argument('-d', '--dynamic', action='store_true', help='Dashboard dynamique (défaut)')
    parser.add_argument('-s', '--static', action='store_true', help='Dashboard statique')
    parser.add_argument('-t', '--text', action='store_true', help='Dashboard texte')
    args = parser.parse_args()
    if args.static:
        dashboard_type = "static"
    elif args.text:
        dashboard_type = "text"
    else:
        dashboard_type = args.type
    print("=" * 80)
    print("OUVERTURE DU DASHBOARD MONITORING MLOPS".center(80))
    print("=" * 80)
    print()
    open_dashboard(dashboard_type)
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()