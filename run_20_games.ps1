param(
    [string]$Username = "JorelMorell"
)

$securePassword = Read-Host "Password" -AsSecureString
$ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
try {
    $password = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
}
finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
}

try {
    python play_public_synthetic.py `
        --username $Username `
        --password $password `
        --team_dir public_gen3ou_teams `
        --battles 20 `
        --checkpoint 48 `
        --enable-battle-ai `
        --log-dir battle_data\ladder20 `
        --database battle_data\ladder20\battles.db `
        --transport-debug `
        --transport-trace battle_data\ladder20\transport_trace.jsonl `
        --analysis
}
finally {
    Remove-Variable password -ErrorAction SilentlyContinue
}
