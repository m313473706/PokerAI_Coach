; Poker AI Coach Windows installer
#define AppName "Poker AI Coach"
#define AppVersion "1.0.0"
#define AppExe "PokerAI_Coach.exe"

[Setup]
AppId={{3D8CA87B-0E0D-43C8-A85B-0DFDA50B89DA}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Poker AI Coach
DefaultDirName={localappdata}\Programs\Poker AI Coach
DefaultGroupName=Poker AI Coach
PrivilegesRequired=lowest
OutputDir=release
OutputBaseFilename=PokerAI_Coach_Ultimate_Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
UninstallDisplayIcon={app}\{#AppExe}
SetupLogging=yes
CloseApplications=yes

[Languages]
Name: "chinesesimp"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "其他选项："; Flags: unchecked

[Files]
Source: "dist\{#AppExe}"; DestDir: "{app}"; Flags: ignoreversion
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "INSTALL.txt"; DestDir: "{app}"; DestName: "安装说明.txt"; Flags: ignoreversion

[Icons]
Name: "{group}\Poker AI Coach"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\Poker AI Coach"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "启动 Poker AI Coach"; Flags: postinstall nowait skipifsilent
