import subprocess
import sys
import platform


class OpenSSHInstaller:
    def __init__(self):
        pass

    def check_command(self, command):
        """
        Check if a command exists on the system.

        parameters
        ----------
        command : str
            The command to check.

        returns
        -------
        bool
            True if the command exists, False otherwise.

        raises
        ------
        subprocess.CalledProcessError
            If the command does not exist.
        """
        try:
            subprocess.run(
                [command, "--version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def install_homebrew(self):
        """
        Install Homebrew if it is not installed.

        parameters
        ----------
        None

        executes
        --------
        subprocess.run
            Install Homebrew.

        raises
        ------
        subprocess.CalledProcessError
            If the installation fails.

        """
        if not self.check_command("brew"):
            print("Homebrew not found. Installing Homebrew...")
            try:
                subprocess.run(
                    [
                        "/bin/bash",
                        "-c",
                        "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)",
                    ],
                    check=True,
                )
                print("Homebrew installed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error installing Homebrew: {e}")
                sys.exit(1)
        else:
            print("Homebrew is already installed.")

    def install_chocolatey(self):
        """
        Install Chocolatey if it is not installed.

        parameters
        ----------
        None

        executes
        --------
        subprocess.run
            Install Chocolatey.

        raises
        ------
        subprocess.CalledProcessError
            If the installation fails.
        """
        if not self.check_command("choco"):
            print("Chocolatey not found. Installing Chocolatey...")
            try:
                subprocess.run(
                    [
                        "powershell",
                        "Set-ExecutionPolicy",
                        "Bypass",
                        "-Scope",
                        "Process",
                        "-Force",
                        ";",
                        "[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072;",
                        "iex",
                        "((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))",
                    ],
                    check=True,
                )
                print("Chocolatey installed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error installing Chocolatey: {e}")
                sys.exit(1)
        else:
            print("Chocolatey is already installed.")

    def install_openssh(self):
        """
        Install OpenSSH if it is not installed.

        parameters
        ----------
        None

        executes
        --------
        if macOS
            subprocess.run
                Install OpenSSH using Homebrew.

        if Windows
            subprocess.run
                Install OpenSSH using Chocolatey.

        raises
        ------
        subprocess.CalledProcessError
            If the installation fails.
        """
        # macOS specific installation.

        if platform.system() == "Darwin":

            self.install_homebrew()
            if not self.check_command("ssh"):
                print("OpenSSH not found. Installing OpenSSH...")
                try:
                    subprocess.run(["brew", "install", "openssh"], check=True)
                    print("OpenSSH installed successfully.")
                except subprocess.CalledProcessError as e:
                    print(f"Error installing OpenSSH: {e}")
                    sys.exit(1)
            else:
                print("OpenSSH is already installed.")
                result = subprocess.run(["ssh", "-V"], capture_output=True, text=True)
                print(f"OpenSSH version: {result.stderr.strip()}")

        # Windows specific installation

        elif platform.system() == "Windows":

            self.install_chocolatey()
            if not self.check_command("ssh"):
                print("OpenSSH not found. Installing OpenSSH...")
                try:
                    subprocess.run(["choco", "install", "openssh", "-y"], check=True)
                    print("OpenSSH installed successfully.")
                except subprocess.CalledProcessError as e:
                    print(f"Error installing OpenSSH: {e}")
                    sys.exit(1)
            else:
                print("OpenSSH is already installed.")
                result = subprocess.run(["ssh", "-V"], capture_output=True, text=True)
                print(f"OpenSSH version: {result.stdout.strip()}")

        else:
            print("Unsupported operating system.")
            sys.exit(1)
