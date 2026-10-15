// SPDX-License-Identifier: Apache-2.0
pragma solidity ^0.8.20;

/**
 * @title VoiceIntegrityLedger
 * @notice EVM-compatible Smart Contract for anchoring tamper-evident Merkle roots
 *         of real-time voice cloning and synthetic speech verification logs.
 * @dev Compatible with Ethereum, Polygon, Arbitrum, Base, and Enterprise Hyperledger Besu.
 */
contract VoiceIntegrityLedger {
    address public owner;

    enum RiskLevel {
        AUTHENTIC_HUMAN,     // 0: Risk < 35%
        ELEVATED_SUSPICIOUS, // 1: 35% <= Risk < 70%
        CRITICAL_CLONE       // 2: Risk >= 70%
    }

    struct CallVerificationRecord {
        bytes32 callSessionId;       // Unique UUID/hash of call session
        bytes32 merkleRoot;          // 32-byte Merkle root of all chunk logs
        bytes32 callerIdHash;        // SHA-256 pseudonymized caller ID
        RiskLevel finalRiskLevel;    // Final classification
        uint16 riskScoreBps;         // Basis points (e.g., 9820 = 98.20%)
        uint32 totalFramesAnalyzed;  // Total sliding chunks processed
        uint256 timestamp;           // Block timestamp of anchoring
        address validator;           // SBC gateway or authorized node
    }

    // Mapping from Call Session ID to On-Chain Cryptographic Record
    mapping(bytes32 => CallVerificationRecord) public records;

    // Authorized voice security gateways
    mapping(address => bool) public authorizedGateways;

    // Events
    event GatewayAuthorized(address indexed gateway);
    event GatewayRevoked(address indexed gateway);
    event CallVerificationAnchored(
        bytes32 indexed callSessionId,
        bytes32 indexed merkleRoot,
        RiskLevel finalRiskLevel,
        uint16 riskScoreBps,
        uint32 totalFrames,
        uint256 timestamp
    );
    event TamperAuditVerified(
        bytes32 indexed callSessionId,
        bytes32 indexed leafHash,
        bool isValid
    );

    modifier onlyOwner() {
        require(msg.sender == owner, "Caller is not the contract owner");
        _;
    }

    modifier onlyAuthorized() {
        require(msg.sender == owner || authorizedGateways[msg.sender], "Unauthorized voice security gateway");
        _;
    }

    constructor() {
        owner = msg.sender;
        authorizedGateways[msg.sender] = true;
    }

    function setGatewayAuthorization(address gateway, bool authorized) external onlyOwner {
        authorizedGateways[gateway] = authorized;
        if (authorized) {
            emit GatewayAuthorized(gateway);
        } else {
            emit GatewayRevoked(gateway);
        }
    }

    /**
     * @notice Anchor cryptographic proof of an evaluated call session onto the blockchain.
     */
    function recordCallVerification(
        bytes32 callSessionId,
        bytes32 merkleRoot,
        bytes32 callerIdHash,
        RiskLevel finalRiskLevel,
        uint16 riskScoreBps,
        uint32 totalFramesAnalyzed
    ) external onlyAuthorized {
        require(records[callSessionId].timestamp == 0, "Call session proof already anchored");
        require(merkleRoot != bytes32(0), "Invalid Merkle root");

        records[callSessionId] = CallVerificationRecord({
            callSessionId: callSessionId,
            merkleRoot: merkleRoot,
            callerIdHash: callerIdHash,
            finalRiskLevel: finalRiskLevel,
            riskScoreBps: riskScoreBps,
            totalFramesAnalyzed: totalFramesAnalyzed,
            timestamp: block.timestamp,
            validator: msg.sender
        });

        emit CallVerificationAnchored(
            callSessionId,
            merkleRoot,
            finalRiskLevel,
            riskScoreBps,
            totalFramesAnalyzed,
            block.timestamp
        );
    }

    /**
     * @notice Cryptographically verify if a specific 250ms audio chunk verification
     *         log was part of the sealed call session using its Merkle Proof.
     * @param callSessionId Session identifier
     * @param leafHash SHA-256 hash of the specific chunk log entry
     * @param proof Array of sibling hashes forming the Merkle proof
     */
    function verifyLogProof(
        bytes32 callSessionId,
        bytes32 leafHash,
        bytes32[] calldata proof
    ) external view returns (bool) {
        bytes32 expectedRoot = records[callSessionId].merkleRoot;
        if (expectedRoot == bytes32(0)) {
            return false;
        }

        bytes32 computedHash = leafHash;
        for (uint256 i = 0; i < proof.length; i++) {
            bytes32 proofElement = proof[i];
            if (computedHash <= proofElement) {
                computedHash = sha256(abi.encodePacked(computedHash, proofElement));
            } else {
                computedHash = sha256(abi.encodePacked(proofElement, computedHash));
            }
        }

        return computedHash == expectedRoot;
    }

    /**
     * @notice Retrieve the full verification record for a call session.
     */
    function getCallRecord(bytes32 callSessionId) external view returns (CallVerificationRecord memory) {
        require(records[callSessionId].timestamp != 0, "No record found for session");
        return records[callSessionId];
    }
}
